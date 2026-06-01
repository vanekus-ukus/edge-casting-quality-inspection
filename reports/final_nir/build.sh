#!/usr/bin/env bash
set -e
xelatex -interaction=nonstopmode main.tex
bibtex main || true
xelatex -interaction=nonstopmode main.tex
xelatex -interaction=nonstopmode main.tex
cp main.pdf final_nir.pdf

