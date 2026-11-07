# BC-Score: Backbone-Conditioned Sidechain Reconstruction for Prioritizing De Novo Protein Binders

Official repository accompanying the paper:
**"Backbone-conditioned sidechain reconstruction improves prioritization of de novo protein binders"**

`BC-Score` strips complex-packed sidechain coordinates from candidate binder–target complexes, reconstructs monomer-conditioned sidechain centroids, distal functional tips, and $\chi_1$ rotamer Mixture Density Network (MDN) distributions using a crystal-pretrained $\text{SE}(3)$-equivariant graph neural network, and combines 38 within-target and 29 cross-target descriptors across five physical modules (`C1`–`C5`) without benchmark-fitted feature weights.

---

## Repository Structure

```text
bc-score/
├── README.md                  # Documentation and reproduction instructions
├── requirements.txt           # Python dependencies
├── reproduce_all.sh           # End-to-end reproduction script (benchmarks + figures + paper PDF)
├── data/
│   ├── overath/
│   │   └── final_dataset.csv  # Overath et al. (2025) benchmark metadata & co-folding baselines (N=3,532 across 13 targets)
│   ├── nipah/
│   │   └── nipah.csv          # Adaptyv Bio / Proteinbase Nipah-G campaign metadata & Boltz-2 baselines (N=881)
│   └── scorpion/
│       ├── benchmark.json     # Anthropic / Adaptyv Scorpion neurotoxin campaign metadata & labels (N=243 across 5 targets)
│       ├── source/
│       │   └── structures_index.csv
│       └── structures/        # All 243 raw pre-co-fold design PDBs (BoltzGen, BindCraft, RFdiffusion)
├── results/
│   ├── locked_v7_model_manifest.json             # Frozen Overath normalization parameters & SHA256 digests
│   ├── backbone_inverse_folding_gnn.pt           # Pretrained Stage B 3D Backbone Inverse-Folding GNN
│   ├── general_protein_sc_and_ebm.pt             # Pretrained Stage C 2-Point SE(3) Sidechain GNN, MDN & Interface Potential
│   ├── singlepose_hardneg_orientation_ebm.pt     # Pretrained Stage A Orientation EBM
│   ├── crystal_corpus_900_slim.pt                # Held-out crystal monomer/complex validation & pretraining cache
│   ├── sabdab2_cluster_reps_cache.pt             # Non-redundant SAbDab crystal cluster representatives
│   ├── overath_5bead_struct_cache.pt             # 5-bead backbone + supplied sidechain coordinates (N=3,532)
│   ├── overath_ebm_features_cache.pt             # Extracted physical descriptors for Overath benchmark (N=3,532)
│   ├── overath_100_timing_cache.pt               # 100-complex Overath runtime cache
│   ├── external_nipah_features_cache.pt          # Extracted physical descriptors for Nipah-G campaign (N=881)
│   ├── external_scorpion_features_cache.pt       # Extracted physical descriptors for Scorpion campaign (N=243)
│   ├── candidate_oof_predictions.csv             # Candidate-level LOTO out-of-fold predictions on Overath (N=3,532)
│   ├── external_nipah_locked_predictions.csv     # Candidate-level locked predictions on Nipah-G (N=881)
│   ├── external_scorpion_locked_predictions.csv  # Candidate-level locked predictions on Scorpion (N=243)
│   ├── loto_benchmark_results.json               # Full LOTO benchmark, ablations, cascades & bootstrap statistics (Tables 1 & Ext. Data 1–3)
│   ├── external_and_runtime_results.json         # Full external validation & A100/CPU runtime results (Ext. Data Table 4 & Fig. 5)
│   └── true_ebm_codesign_results.json            # M=1 True-EBM & 39-scaffold partition-function co-design results (Ext. Data Fig. 2)
├── scripts/
│   ├── stage_a_orientation.py        # Stage A: SE(3) monomer encoder, orientation EBM & ESM-2 frustration
│   ├── stage_b_foldability.py        # Stage B: 3D backbone inverse-folding GNN & monomer physics
│   ├── stage_c_sidechain_ebm.py      # Stage C: 2-Point sidechain GNN, rotamer MDN, interface potential & 24-pose funnel
│   ├── codesign_engine.py            # Sequence + 6-DoF pose co-design engine
│   ├── benchmark_batched_gpu.py      # Disjoint-graph batched A100 GPU & CPU pipeline evaluator
│   ├── run_loto_benchmark.py         # Reproduces 13-fold LOTO benchmark, ablations, cascades & sequence-identity audit
│   ├── run_external_and_runtime.py   # Reproduces held-out Nipah-G & Scorpion external validation & runtime tables
│   ├── run_true_ebm_and_codesign.py  # Reproduces M=1 True-EBM & 39-scaffold partition-function co-design
│   ├── plot_fig1_pipeline.py         # Generates Figure 1 (fig1_pipeline_and_mdm2.pdf)
│   ├── plot_fig2_fig3_benchmark.py   # Generates Figure 2 (fig2_loto_benchmark.pdf) & Figure 3 (fig3_sidechain_ablation_and_modules.pdf)
│   ├── plot_fig4_fig5_and_ext.py     # Generates Figure 4 (fig4_cascade_and_hybrids.pdf), Figure 5 (fig5_external_and_runtime.pdf) & Ext. Data Figs. 1–2
│   ├── plot_ext_fig3_epitope.py      # Generates Extended Data Figure 3 (ext_fig3_same_epitope_pairs.pdf)
│   └── render_pymol_mdm2.py          # PyMOL 3D structure renderer for Figure 1b,c Mdm2 interfaces
└── paper/
    ├── main.tex                      # LaTeX manuscript source
    ├── references.bib                # BibTeX bibliography
    ├── main.pdf                      # Compiled manuscript and Extended Data PDF
    └── figures/                      # All main and Extended Data figures (fig1..fig5, ext_fig1..ext_fig3)
```

---

## Quick Start & Reproducing Paper Results

### 1. Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Verify Pretrained Checkpoint & Manifest Integrity

```bash
shasum -a 256 \
  results/locked_v7_model_manifest.json \
  results/general_protein_sc_and_ebm.pt \
  results/backbone_inverse_folding_gnn.pt \
  results/singlepose_hardneg_orientation_ebm.pt \
  results/overath_ebm_features_cache.pt
```

Expected SHA256 digests:
- `093ecc7f56f6865017d8601f7a31f5fe1e83af36ac90008d965e313a873eecc2` — `results/locked_v7_model_manifest.json`
- `ebf1073ea0e20c77ffafcf2a1cbc4b7c8daf4462eea640a1306f99f4dc6d3fe7` — `results/general_protein_sc_and_ebm.pt`
- `97d9a92cee2a1f4fcec5bb4630235ff22d61bb70a3f2f05c1b0f360950a8b932` — `results/backbone_inverse_folding_gnn.pt`
- `eb551c376783e62add3f6bd813014cb6922a180603bc15e36b585861be796c08` — `results/singlepose_hardneg_orientation_ebm.pt`
- `f6c27649498cceb551d66d756a17be036b572fda2de578b0ca9a960ad0a19f07` — `results/overath_ebm_features_cache.pt`

### 3. Reproduce All Benchmarks, Figures, and Manuscript PDF

Run the end-to-end reproduction script:
```bash
bash reproduce_all.sh
```

Or run individual steps:
```bash
# 1. 13-fold LOTO benchmark, sidechain attribution ablation, two-stage cascade & sequence-identity audit
python3 scripts/run_loto_benchmark.py

# 2. Held-out external validation (Nipah-G N=881 & Scorpion N=243) and runtime summary
python3 scripts/run_external_and_runtime.py

# 3. Single-candidate (M=1) True-EBM evaluation & 39-scaffold partition-function co-design
python3 scripts/run_true_ebm_and_codesign.py

# 4. Regenerate all main figures (Figures 1–5) and Extended Data Figures 1–3
python3 scripts/plot_fig1_pipeline.py
python3 scripts/plot_fig2_fig3_benchmark.py
python3 scripts/plot_fig4_fig5_and_ext.py
```
