#!/usr/bin/env bash
# Deploy the overnight branch to production.
# Run from the qbcc-site repository root.
#
# Sequence:
#   1. Push the pre-overnight rollback tag so we have a remote rollback point.
#   2. Push the overnight branch as a branch reference on origin.
#   3. Push the overnight branch's HEAD to origin/main, which triggers the
#      Render auto-deploy.
#
# Rollback (run from the repo root if deploy goes badly):
#   git push --force origin pre-overnight-2026-05-10:main
#
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

echo ">>> Pushing rollback tag..."
git push origin pre-overnight-2026-05-10

echo ">>> Pushing overnight branch reference..."
git push origin overnight/app-eastbrook-pass:overnight/app-eastbrook-pass

echo ">>> Pushing branch HEAD to main (triggers Render deploy)..."
git push origin overnight/app-eastbrook-pass:main

echo ""
echo "Done. Render auto-deploys when origin/main updates."
echo "Watch the deploy at https://dashboard.render.com/."
echo ""
echo "To roll back if anything is broken:"
echo "  git push --force origin pre-overnight-2026-05-10:main"
