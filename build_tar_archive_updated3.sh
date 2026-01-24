#!/usr/bin/env bash
set -euo pipefail

# Create a code-only tarball for the repo, excluding large outputs and *all* common env / local config files.
# Run from the repository root.
#
# NOTE: tar's --exclude matching is a bit picky about leading "./".
# These patterns are written to match BOTH ".venv/..." and "./.venv/..." forms.

OUT="esbjerg2_publication_code_only.tar.gz"

tar -czvf "${OUT}" \
  --exclude='.venv' \
  --exclude='.venv/**' \
  --exclude='./.venv' \
  --exclude='./.venv/**' \
  --exclude='venv' \
  --exclude='venv/**' \
  --exclude='./venv' \
  --exclude='./venv/**' \
  --exclude='.env' \
  --exclude='.env.*' \
  --exclude='.envrc' \
  --exclude='.Renviron' \
  --exclude='.Renviron.*' \
  --exclude='.Rprofile' \
  --exclude='.Rhistory' \
  --exclude='.RData' \
  --exclude='.Ruserdata' \
  --exclude='.Rproj.user' \
  --exclude='.Rproj.user/**' \
  --exclude='./.Rproj.user' \
  --exclude='./.Rproj.user/**' \
  --exclude='.python-version' \
  --exclude='.tool-versions' \
  --exclude='.vscode' \
  --exclude='.vscode/**' \
  --exclude='.idea' \
  --exclude='.idea/**' \
  --exclude='__pycache__' \
  --exclude='**/__pycache__/**' \
  --exclude='*.pyc' \
  --exclude='renv/library' \
  --exclude='renv/library/**' \
  --exclude='renv/python' \
  --exclude='renv/python/**' \
  --exclude='renv/staging' \
  --exclude='renv/staging/**' \
  --exclude='OUTPUT' \
  --exclude='OUTPUT/**' \
  --exclude='RESULTS' \
  --exclude='RESULTS/**' \
  --exclude='FIGURES' \
  --exclude='FIGURES/**' \
  --exclude='OLD' \
  --exclude='OLD/**' \
  --exclude='CODEOCEAN' \
  --exclude='CODEOCEAN/**' \
  --exclude='*.rds' \
  --exclude='*.png' \
  --exclude='*.tar' \
  --exclude='*.tar.gz' \
  --exclude='*.tgz' \
  .

echo "Created: ${OUT}"
echo "Tip: verify excludes with: tar -tzf ${OUT} | head"
