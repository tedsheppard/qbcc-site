#!/usr/bin/env bash
# Sopal v2 — fetch the curated public test fixtures.
# See legal_corpus/test_fixtures/README.md for what this pulls and why.
# Idempotent: skips anything already on disk.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FIX="$ROOT/legal_corpus/test_fixtures"

# Helper: download $1 to $2 only if $2 doesn't already exist.
download() {
  local url="$1"
  local dest="$2"
  local label="$3"
  if [[ -s "$dest" ]]; then
    echo "  [skip] $label — already present at $dest"
    return 0
  fi
  mkdir -p "$(dirname "$dest")"
  echo "  [get]  $label"
  if ! curl -fL --connect-timeout 15 --max-time 90 -A "Sopal-v2-fixtures/1.0 (test data)" -o "$dest" "$url"; then
    echo "  [fail] $label — could not download from $url"
    rm -f "$dest"
    return 1
  fi
  if [[ ! -s "$dest" ]]; then
    echo "  [fail] $label — downloaded but file is empty"
    rm -f "$dest"
    return 1
  fi
  echo "  [ok]   $label — $(wc -c <"$dest" | tr -d ' ') bytes"
}

# Helper: write a per-fixture source.txt note (always, even on skip).
write_source_note() {
  local dir="$1"
  local url="$2"
  local desc="$3"
  local licence="$4"
  mkdir -p "$dir"
  cat >"$dir/source.txt" <<EOF
Source URL: $url
Description: $desc
Retrieved: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
Licence: $licence
EOF
}

echo "Sopal v2 — public-contract test fixtures"
echo "Destination: $FIX"
echo

# Note: URLs below were discovered by inspecting the canonical landing
# pages on 2026-05-09. CMS-managed file paths rotate occasionally — re-run
# `WebFetch` against the source URL in source.txt if a [fail] reappears.

# --- 1. QBCC Payment Claim template (QLD) ----------------------------------
DEST="$FIX/qbcc-pc-template"
write_source_note "$DEST" \
  "https://www.qbcc.qld.gov.au/resources/template/payment-claim-template" \
  "QBCC Payment Claim template (BIF Act compliant)" \
  "QLD Government / QBCC. Crown copyright; published for industry use under BIF Act."
echo "[1/4] QBCC Payment Claim template"
download \
  "https://www.qbcc.qld.gov.au/sites/default/files/2021-09/template-bif-payment-claim.pdf" \
  "$DEST/template-bif-payment-claim.pdf" \
  "QBCC PC template (.pdf)" || true

# --- 2. NSW GC21 Edition 2 (NSW) -------------------------------------------
DEST="$FIX/nsw-gc21-ed2"
write_source_note "$DEST" \
  "https://www.info.buy.nsw.gov.au/resources/gc21" \
  "NSW Government — GC21 Edition 2 head contract" \
  "NSW Government / Crown. Published for re-use; treat as Crown licence."
echo "[2/4] NSW GC21 Ed.2"
download \
  "https://www.info.buy.nsw.gov.au/__data/assets/word_doc/0008/613295/gc21-edition-2-general-conditions-of-contract-070426.docx" \
  "$DEST/gc21-general-conditions-of-contract.docx" \
  "GC21 General Conditions of Contract" || true
download \
  "https://www.info.buy.nsw.gov.au/__data/assets/word_doc/0008/617489/gc21-edition-2-preliminaries.docx" \
  "$DEST/gc21-preliminaries.docx" \
  "GC21 Preliminaries" || true
download \
  "https://www.info.buy.nsw.gov.au/__data/assets/word_doc/0007/617821/GC21-edition-2-clause-commentary_25-02-2025.docx" \
  "$DEST/gc21-clause-commentary.docx" \
  "GC21 Clause Commentary" || true

# --- 3. NSW MW21 — local government variant (NSW) --------------------------
DEST="$FIX/nsw-mw21"
write_source_note "$DEST" \
  "https://www.info.buy.nsw.gov.au/resources/mw21localgov" \
  "NSW Government — MW21 Minor Works (local government variant)" \
  "NSW Government / Crown. Published for re-use; treat as Crown licence."
echo "[3/4] NSW MW21 (local govt variant)"
download \
  "https://www.info.buy.nsw.gov.au/__data/assets/word_doc/0006/1281309/MW21-LG_General-Conditions_23_10_2025.docx" \
  "$DEST/mw21-lg-general-conditions.docx" \
  "MW21-LG General Conditions" || true
download \
  "https://www.info.buy.nsw.gov.au/__data/assets/pdf_file/0003/1281441/mw21_clause-commentary_15-02-2024.pdf" \
  "$DEST/mw21-clause-commentary.pdf" \
  "MW21 Clause Commentary" || true

# --- 4. ACT MW21 General Conditions (ACT) ---------------------------------
DEST="$FIX/act-mw21"
write_source_note "$DEST" \
  "https://www.act.gov.au/infrastructurecanberra/supplying-to-government/supplier-information/infrastructure-contract-suite/mw21-document-suite" \
  "ACT Government — MW21 General Conditions of Contract" \
  "ACT Government / Crown. Published for industry use."
echo "[4/4] ACT MW21 General Conditions"
download \
  "https://www.act.gov.au/__data/assets/pdf_file/0011/1942733/MW21-General-Conditions-of-Contract.pdf" \
  "$DEST/mw21-general-conditions-of-contract.pdf" \
  "ACT MW21 General Conditions of Contract" || true

echo
echo "Done. Inspect $FIX/ — anything that didn't download will print '[fail]' above."
echo "Re-run safely; existing files are skipped."
