#!/usr/bin/env bash
# Download QBCC decision PDFs from a list of "app|url" lines on stdin.
# Skips files that already exist and are non-empty.
set -u
DIR="${DIR:-_local_data/new_decisions_2026-05-10}"
mkdir -p "$DIR"
pids=()
while IFS='|' read -r app url; do
  [ -z "$app" ] && continue
  out="$DIR/$app.pdf"
  if [ -s "$out" ]; then
    echo "skip $app (exists)"
    continue
  fi
  (curl -sSL -o "$out" "$url" -w "%{http_code} $app\n" || echo "FAIL $app") &
  pids+=($!)
done
wait
