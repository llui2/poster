#!/usr/bin/env bash
set -euo pipefail

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi

.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python figures/make_figures.py

if ! command -v lualatex >/dev/null 2>&1; then
  echo "lualatex not found. Install a TeX distribution (e.g. MacTeX) and rerun ./build.sh." >&2
  exit 1
fi

lualatex -interaction=nonstopmode -halt-on-error poster.tex
lualatex -interaction=nonstopmode -halt-on-error poster.tex
