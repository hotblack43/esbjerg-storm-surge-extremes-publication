#!/usr/bin/env bash
set -euo pipefail

RMD="${1:?Usage: $0 path/to/file.Rmd [optional args...]}"
shift || true

# Root knitting at the Rmd's folder so DATA/, OUTPUT/ etc resolve correctly.
RMD_DIR="$(cd "$(dirname "$RMD")" && pwd)"
RMD_BASENAME="$(basename "$RMD")"

# Run from the Rmd directory; pass any extra args after -- (available as commandArgs(trailingOnly=TRUE) in R)
cd "$RMD_DIR"

Rscript -e "rmarkdown::render('$RMD_BASENAME', knit_root_dir='$RMD_DIR')" -- "$@"

