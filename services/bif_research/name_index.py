"""Deterministic name -> chunk-id lookups for cases and provisions.

The knowledge-augmented planner emits the *names* of authorities and
provisions ("MWB Everton Park v Devcon", "BIF Act s 75"). This module
resolves those names to concrete chunk_ids in the corpus so the retriever
can pull them directly, bypassing BM25/dense scoring noise.

Two indexes, both stored in `store/name_index.sqlite`:

  case_index
    case_id          TEXT PRIMARY KEY   (e.g. "QCA_2024_94")
    case_name        TEXT               (canonical, e.g. "MWB Everton Park ...")
    citation         TEXT               ("MWB Everton Park ... [2024] QCA 94")
    court            TEXT               ("QCA")
    year             TEXT               ("2024")
    case_number      TEXT               ("94")
    norm_name        TEXT               (lowercased, punct-stripped)
    chunk_count      INTEGER

  case_chunks
    case_id          TEXT
    chunk_id         TEXT
    paragraph_start  INTEGER
    paragraph_end    INTEGER
    rank_within_case INTEGER            (chunk order within the judgment)

  provision_index
    provision_key    TEXT PRIMARY KEY   (e.g. "bif_act_s75")
    act_short        TEXT               ("BIF Act")
    act_name         TEXT               (full long title)
    section_number   TEXT               ("75")
    section_title    TEXT               ("Time when payment claim must be paid")
    chunk_count      INTEGER

  provision_chunks
    provision_key    TEXT
    chunk_id         TEXT
    rank_within_section INTEGER

Lookup functions are case-insensitive, punctuation-tolerant, and tolerate
common variations (e.g. "section 75", "s 75", "s75", "BIFA s 75").
"""
from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CHUNKS_DB = ROOT / "store" / "chunks.sqlite"
NAME_DB = ROOT / "store" / "name_index.sqlite"

# Map raw act_short -> canonical short key used in provision_key
ACT_SHORT_TO_KEY = {
    "BIF Act":  "bif_act",
    "BIF Reg":  "bif_reg",
    "QBCC Act": "qbcc_act",
    "QBCC Reg": "qbcc_reg",
    "AIA":      "aia",
}

# Reverse for prettier display
KEY_TO_DISPLAY = {v: k for k, v in ACT_SHORT_TO_KEY.items()}

# Common case-name boilerplate to strip when normalising
_NAME_NOISE = re.compile(
    r"\b(pty|ltd|p\/l|plc|inc|corp|corporation|limited|proprietary|"
    r"the|and|of|in|liquidation|liq|administrators?|appointed|receivers?)\b",
    re.I,
)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _norm_name(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)  # strip punctuation
    s = _NAME_NOISE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _norm_section(num: str) -> str:
    return re.sub(r"[^\w]", "", str(num)).lower()


def _strip_para_suffix(source_id: str) -> str:
    """`QCA_2001_433_p1-4_c0` -> `QCA_2001_433`."""
    # Pattern is COURT_YEAR_NUMBER then optional _p..._c... or similar
    parts = source_id.split("_")
    # Walk until we hit the first part starting with 'p' followed by a digit
    keep = []
    for p in parts:
        if re.match(r"^p\d", p) or re.match(r"^c\d", p):
            break
        keep.append(p)
    return "_".join(keep)


def build(verbose: bool = True) -> dict:
    """Scan chunks.sqlite and (re)build the name_index.sqlite indexes.

    Idempotent — drops and recreates tables. Returns a stats dict.
    """
    if not CHUNKS_DB.exists():
        raise FileNotFoundError(f"chunks DB missing: {CHUNKS_DB}")

    src = sqlite3.connect(CHUNKS_DB)
    dst = sqlite3.connect(NAME_DB)
    cur = dst.cursor()
    cur.executescript(
        """
        DROP TABLE IF EXISTS case_index;
        DROP TABLE IF EXISTS case_chunks;
        DROP TABLE IF EXISTS provision_index;
        DROP TABLE IF EXISTS provision_chunks;

        CREATE TABLE case_index (
            case_id      TEXT PRIMARY KEY,
            case_name    TEXT,
            citation     TEXT,
            court        TEXT,
            year         TEXT,
            case_number  TEXT,
            norm_name    TEXT,
            chunk_count  INTEGER
        );
        CREATE INDEX idx_case_norm ON case_index(norm_name);

        CREATE TABLE case_chunks (
            case_id           TEXT,
            chunk_id          TEXT,
            paragraph_start   INTEGER,
            paragraph_end     INTEGER,
            rank_within_case  INTEGER,
            PRIMARY KEY (case_id, chunk_id)
        );
        CREATE INDEX idx_case_chunks_case ON case_chunks(case_id);

        CREATE TABLE provision_index (
            provision_key   TEXT PRIMARY KEY,
            act_short       TEXT,
            act_name        TEXT,
            section_number  TEXT,
            section_title   TEXT,
            chunk_count     INTEGER
        );
        CREATE INDEX idx_prov_section ON provision_index(section_number);

        CREATE TABLE provision_chunks (
            provision_key       TEXT,
            chunk_id            TEXT,
            rank_within_section INTEGER,
            PRIMARY KEY (provision_key, chunk_id)
        );
        CREATE INDEX idx_prov_chunks_key ON provision_chunks(provision_key);
        """
    )

    # ---- cases ----
    cases: dict[str, dict] = {}
    case_chunks: dict[str, list[tuple]] = {}
    rows = src.execute(
        "SELECT chunk_id, source_id, metadata_json FROM chunks WHERE source_type='judgment'"
    ).fetchall()
    for chunk_id, source_id, meta_json in rows:
        try:
            m = json.loads(meta_json or "{}")
        except json.JSONDecodeError:
            m = {}
        case_id = _strip_para_suffix(source_id)
        if not case_id:
            continue
        if case_id not in cases:
            cases[case_id] = {
                "case_id": case_id,
                "case_name": m.get("case_name", ""),
                "citation": m.get("citation", ""),
                "court": m.get("court", ""),
                "year": m.get("year", ""),
                "case_number": m.get("case_number", ""),
                "norm_name": _norm_name(m.get("case_name", "")),
            }
            case_chunks[case_id] = []
        case_chunks[case_id].append((
            chunk_id,
            int(m.get("paragraph_start", 0) or 0),
            int(m.get("paragraph_end", 0) or 0),
        ))

    for case_id, info in cases.items():
        chunks = sorted(case_chunks[case_id], key=lambda x: (x[1], x[2], x[0]))
        cur.execute(
            """INSERT INTO case_index
               (case_id, case_name, citation, court, year, case_number, norm_name, chunk_count)
               VALUES (?,?,?,?,?,?,?,?)""",
            (info["case_id"], info["case_name"], info["citation"], info["court"],
             info["year"], info["case_number"], info["norm_name"], len(chunks)),
        )
        for rank, (cid, ps, pe) in enumerate(chunks):
            cur.execute(
                "INSERT INTO case_chunks VALUES (?,?,?,?,?)",
                (case_id, cid, ps, pe, rank),
            )

    # ---- provisions (statute + regulation only — annotated is non-citable) ----
    prov: dict[str, dict] = {}
    prov_chunks: dict[str, list[str]] = {}
    rows = src.execute(
        "SELECT chunk_id, metadata_json FROM chunks "
        "WHERE source_type IN ('statute','regulation')"
    ).fetchall()
    for chunk_id, meta_json in rows:
        try:
            m = json.loads(meta_json or "{}")
        except json.JSONDecodeError:
            continue
        act_short = m.get("act_short", "")
        section_number = m.get("section_number", "")
        if not act_short or not section_number:
            continue
        act_key = ACT_SHORT_TO_KEY.get(act_short)
        if not act_key:
            continue
        provision_key = f"{act_key}_s{_norm_section(section_number)}"
        if provision_key not in prov:
            prov[provision_key] = {
                "provision_key": provision_key,
                "act_short": act_short,
                "act_name": m.get("act_name", ""),
                "section_number": section_number,
                "section_title": m.get("section_title", ""),
            }
            prov_chunks[provision_key] = []
        prov_chunks[provision_key].append(chunk_id)

    for key, info in prov.items():
        chunks = sorted(prov_chunks[key])  # stable order: chunk_id sort = corpus order
        cur.execute(
            """INSERT INTO provision_index
               (provision_key, act_short, act_name, section_number, section_title, chunk_count)
               VALUES (?,?,?,?,?,?)""",
            (info["provision_key"], info["act_short"], info["act_name"],
             info["section_number"], info["section_title"], len(chunks)),
        )
        for rank, cid in enumerate(chunks):
            cur.execute(
                "INSERT INTO provision_chunks VALUES (?,?,?)",
                (key, cid, rank),
            )

    dst.commit()
    stats = {
        "n_cases": len(cases),
        "n_case_chunks": sum(len(v) for v in case_chunks.values()),
        "n_provisions": len(prov),
        "n_provision_chunks": sum(len(v) for v in prov_chunks.values()),
    }
    if verbose:
        print(f"name_index built: {stats}")
    src.close()
    dst.close()
    return stats


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------

@dataclass
class ProvisionMatch:
    provision_key: str
    act_short: str
    section_number: str
    section_title: str
    chunk_ids: list[str]
    confidence: str  # "exact" | "fuzzy"


@dataclass
class CaseMatch:
    case_id: str
    case_name: str
    citation: str
    court: str
    year: str
    chunk_ids: list[str]
    confidence: str  # "exact" | "fuzzy"


_conn: sqlite3.Connection | None = None


def _db() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        if not NAME_DB.exists():
            raise FileNotFoundError(
                f"name_index.sqlite missing — run `python3 -m services.bif_research.name_index build`"
            )
        _conn = sqlite3.connect(NAME_DB)
        _conn.row_factory = sqlite3.Row
    return _conn


# ---- provision parsing ----

# Map common shorthand the planner / user might write to act_short
_ACT_NAME_PATTERNS = [
    (re.compile(r"\bbif\s*act\b|\bbifa\b|\bbuilding\s+industry\s+fairness", re.I), "BIF Act"),
    (re.compile(r"\bbif\s*reg\b|\bbifr\b|\bbuilding\s+industry\s+fairness.*regulation", re.I), "BIF Reg"),
    (re.compile(r"\bqbcc\s*act\b|\bqueensland\s+building\s+and\s+construction\s+commission\s+act", re.I), "QBCC Act"),
    (re.compile(r"\bqbcc\s*reg\b|\bqueensland\s+building\s+and\s+construction\s+commission\s+regulation", re.I), "QBCC Reg"),
    (re.compile(r"\bacts?\s+interpretation\s+act\b|\baia\b", re.I), "AIA"),
]

_SECTION_PATTERN = re.compile(
    r"(?:section|sec|s|sch)\s*\.?\s*([0-9]+[A-Za-z]*(?:\s*\([0-9a-zA-Z]+\))*)",
    re.I,
)


def parse_provision(text: str) -> tuple[str | None, str | None]:
    """Return (act_short, normalised_section) or (None, None)."""
    if not text:
        return None, None
    act_short = None
    for pat, label in _ACT_NAME_PATTERNS:
        if pat.search(text):
            act_short = label
            break
    m = _SECTION_PATTERN.search(text)
    if not m:
        return act_short, None
    sec_raw = m.group(1).strip()
    # Drop subsection brackets — we index whole sections
    sec_main = re.split(r"[(\s]", sec_raw, maxsplit=1)[0]
    return act_short, _norm_section(sec_main)


def lookup_provision(text: str) -> ProvisionMatch | None:
    """Resolve text like "BIF Act s 75" or "section 100 of the QBCC Act"
    to a ProvisionMatch. Returns None if act or section can't be parsed."""
    act_short, section = parse_provision(text)
    if not act_short or not section:
        return None
    act_key = ACT_SHORT_TO_KEY.get(act_short)
    if not act_key:
        return None
    key = f"{act_key}_s{section}"
    db = _db()
    row = db.execute(
        "SELECT * FROM provision_index WHERE provision_key=?", (key,)
    ).fetchone()
    if not row:
        return None
    chunk_rows = db.execute(
        "SELECT chunk_id FROM provision_chunks WHERE provision_key=? "
        "ORDER BY rank_within_section",
        (key,),
    ).fetchall()
    return ProvisionMatch(
        provision_key=key,
        act_short=row["act_short"],
        section_number=row["section_number"],
        section_title=row["section_title"] or "",
        chunk_ids=[r["chunk_id"] for r in chunk_rows],
        confidence="exact",
    )


# ---- case lookup ----

# Common case-name shorthand the planner / lawyer would write
_CASE_CITATION_PATTERN = re.compile(
    r"\[(\d{4})\]\s*(QCA|QSC|QDC|QSCA|QCAT|HCA|NSWCA|NSWSC|FCA|FCAFC)\s*(\d+)",
    re.I,
)


def _tokenise_name(s: str) -> set[str]:
    return set(_norm_name(s).split())


def lookup_case(text: str, *, min_overlap: int = 1) -> CaseMatch | None:
    """Resolve a case name (with or without citation) to a CaseMatch.

    Strategy:
      1. If the text contains a [YEAR] COURT NUMBER citation, look up by
         (court, year, case_number) directly. exact confidence.
      2. Otherwise tokenise the text, fetch all case norm_names, score by
         token overlap (Jaccard-ish), pick best if score >= threshold. fuzzy.
    """
    if not text:
        return None
    db = _db()

    # ---- Path 1: citation match ----
    m = _CASE_CITATION_PATTERN.search(text)
    if m:
        year, court, number = m.group(1), m.group(2).upper(), m.group(3)
        # Try direct case_id match first (case_id format: COURT_YEAR_NUMBER)
        case_id_guess = f"{court}_{year}_{number}"
        row = db.execute(
            "SELECT * FROM case_index WHERE case_id=?", (case_id_guess,)
        ).fetchone()
        if not row:
            # Try by court+year+case_number
            row = db.execute(
                "SELECT * FROM case_index WHERE court=? AND year=? AND case_number=?",
                (court, year, number),
            ).fetchone()
        if row:
            return _materialise_case(db, row, "exact")

    # ---- Path 2: token-overlap fuzzy ----
    query_tokens = _tokenise_name(text)
    # Drop very short / common tokens
    query_tokens = {t for t in query_tokens if len(t) > 2}
    if not query_tokens:
        return None
    best: tuple[float, sqlite3.Row | None] = (0.0, None)
    for row in db.execute("SELECT * FROM case_index").fetchall():
        if not row["norm_name"]:
            continue
        case_tokens = set(row["norm_name"].split())
        case_tokens = {t for t in case_tokens if len(t) > 2}
        if not case_tokens:
            continue
        overlap = query_tokens & case_tokens
        if len(overlap) < min_overlap:
            continue
        # Jaccard with a slight bias toward more-specific case-side tokens
        union = query_tokens | case_tokens
        score = len(overlap) / len(union)
        # Boost if all the rare-side tokens match
        if overlap == case_tokens:
            score += 0.5
        if score > best[0]:
            best = (score, row)

    # Threshold: require at least a moderate overlap
    if best[1] is None or best[0] < 0.15:
        return None
    return _materialise_case(db, best[1], "fuzzy")


def _materialise_case(db: sqlite3.Connection, row: sqlite3.Row, conf: str) -> CaseMatch:
    chunk_rows = db.execute(
        "SELECT chunk_id FROM case_chunks WHERE case_id=? ORDER BY rank_within_case",
        (row["case_id"],),
    ).fetchall()
    return CaseMatch(
        case_id=row["case_id"],
        case_name=row["case_name"] or "",
        citation=row["citation"] or "",
        court=row["court"] or "",
        year=row["year"] or "",
        chunk_ids=[r["chunk_id"] for r in chunk_rows],
        confidence=conf,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2 or sys.argv[1] not in ("build", "lookup"):
        print("usage: python3 -m services.bif_research.name_index build")
        print("       python3 -m services.bif_research.name_index lookup '<text>'")
        sys.exit(1)
    if sys.argv[1] == "build":
        build()
    else:
        text = " ".join(sys.argv[2:])
        p = lookup_provision(text)
        c = lookup_case(text)
        print("provision:", p)
        print("case:", c)
