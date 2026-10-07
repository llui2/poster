#!/usr/bin/env bash
set -euo pipefail

python3 figures/make_figures.py
lualatex -interaction=nonstopmode -halt-on-error poster.tex
lualatex -interaction=nonstopmode -halt-on-error poster.tex
