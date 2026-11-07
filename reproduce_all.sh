#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "=== 1. Verifying SHA256 digests of frozen manifest and checkpoints ==="
shasum -a 256 \
  results/locked_v7_model_manifest.json \
  results/general_protein_sc_and_ebm.pt \
  results/backbone_inverse_folding_gnn.pt \
  results/singlepose_hardneg_orientation_ebm.pt \
  results/overath_ebm_features_cache.pt

echo "=== 2. Reproducing Held-Out External Validation (Nipah-G & Scorpion) ==="
python3 scripts/run_external_and_runtime.py

echo "=== 3. Regenerating All Main Figures (Figures 1-5) and Extended Data Figures 1-3 ==="
python3 scripts/plot_fig1_pipeline.py
python3 scripts/plot_fig2_fig3_benchmark.py
python3 scripts/plot_fig4_fig5_and_ext.py

echo "=== 4. Compiling Manuscript PDF (paper/main.pdf) ==="
if command -v pdflatex >/dev/null 2>&1; then
  (cd paper && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex)
fi

echo "=== All results and figures reproduced successfully! ==="
