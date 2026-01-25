#!/usr/bin/env bash
set -euo pipefail

# Create a code-only tarball for the repo, excluding large outputs and *all* common env / local config files.
# Run from the repository root.

tar -czvf esbjerg2_publication_code_only.tar.gz \
  --exclude='./.venv/' \
  --exclude='./venv/' \
  --exclude='./.env' \
  --exclude='./.env.*' \
  --exclude='./.envrc' \
  --exclude='./.Renviron' \
  --exclude='./.Renviron.*' \
  --exclude='./.Rprofile' \
  --exclude='./.Rhistory' \
  --exclude='./.RData' \
  --exclude='./.Ruserdata' \
  --exclude='./.Rproj.user/' \
  --exclude='./.python-version' \
  --exclude='./.tool-versions' \
  --exclude='./.vscode/' \
  --exclude='./.idea/' \
  --exclude='./__pycache__/' \
  --exclude='**/__pycache__/' \
  --exclude='*.pyc' \
  --exclude='./renv/library/' \
  --exclude='./renv/python/' \
  --exclude='./renv/staging/' \
  --exclude='./OUTPUT' --exclude='./OUTPUT/**' \
  --exclude='./RESULTS' --exclude='./RESULTS/**' \
  --exclude='./FIGURES' --exclude='./FIGURES/**' \
  --exclude='./OLD' \
  --exclude='./CODEOCEAN' \
  --exclude='*.rds' \
  --exclude='*.png' \
  --exclude='*.tar' \
  --exclude='*.tar.gz' \
  --exclude='*.tgz' \
  .
