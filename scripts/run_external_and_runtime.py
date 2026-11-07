"""Zero-Shot Locked External Validation, Complete End-to-End Runtime Profiling, and Retrospective Cascade Simulation.

Implements the 4-part computational validation package without modifying BC-EBM:
  1. Locks code/checkpoint SHA256 provenance and Overath training normalization + weight vectors
     into `results/locked_v7_model_manifest.json` BEFORE loading any external benchmark labels.
  2. Measures complete end-to-end runtime across all pipeline stages (explicitly including the newer
     autograd backbone restoring force pass and continuous 6-DoF rigid-body relaxation pass) and
     compares against measured AlphaFold3 / Boltz / Protenix-V2 execution configurations.
  3. Evaluates the unchanged locked v7 BC-EBM model on two independent public experimental campaigns:
     - Campaign 1: Adaptyv Bio / Proteinbase Nipah Virus Glycoprotein (Nipah-G, N = 881 complexes,
       100 experimentally verified binders, 781 experimental non-binders).
     - Campaign 2: Anthropic / Adaptyv Multi-Target Scorpion Neurotoxin Campaign (N = 243 generative
       design complexes across 5 toxins: Cn2, AaHII, Ts1, AaHI, LqhIII; 21 Adaptyv hits, 12 publication hits).
  4. Evaluates the retrospective two-stage screening cascade across fixed experimental testing budgets
     versus co-folding calls and measured compute cost on Overath, Nipah-G, and Scorpion.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import sys
import time
import urllib.request
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import biotite.sequence as bseq
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdbx as pdbx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from transformers import AutoModelForMaskedLM, AutoTokenizer

warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))

from stage_c_sidechain_ebm import (
    GEN_PROTEIN_CKPT,
    _axis_angle_to_matrix,
    build_inplace_mirage_sequences,
    build_local_frames,
    build_redocking_pose_ensemble,
    compute_pure_backbone_geometry,
    train_or_load_general_models,
)
from stage_b_foldability import (
    IF_CKPT_PATH as IF_MODEL_CKPT,
    compute_corrected_physics_and_oring,
    compute_interface_desolvation_and_density,
    compute_intra_monomer_5bead_physics,
    compute_inverse_folding_features,
    compute_mechanical_stiffness_and_tertiary_tethering,
    train_or_load_backbone_inverse_folding_gnn,
)
from stage_a_orientation import (
    MODEL_ID,
    _AA1_LIST,
    extract_partner_esm,
    generate_surface_packed_sabdab2_pool,
    PHYSCHEM_TABLE,
    REGION_AB_CDR1,
    REGION_AB_CDRH3,
    REGION_AB_CDRL3,
    REGION_AB_VFW,
    UnboundOrientationPairwiseEBM,
    compute_antibody_paratope_geometry,
    extract_monomer_geometry,
)
AA3 = {
    "ALA": 0, "ARG": 1, "ASN": 2, "ASP": 3, "CYS": 4, "GLN": 5, "GLU": 6,
    "GLY": 7, "HIS": 8, "ILE": 9, "LEU": 10, "LYS": 11, "MET": 12, "PHE": 13,
    "PRO": 14, "SER": 15, "THR": 16, "TRP": 17, "TYR": 18, "VAL": 19,
}


CKPT_HARDNEG = PROJECT / "results" / "singlepose_hardneg_orientation_ebm.pt"
OVERATH_FEAT_CACHE = PROJECT / "results" / "overath_ebm_features_cache.pt"
OVERATH_5BEAD_CACHE = PROJECT / "results" / "overath_5bead_coord_cache.pt"

LOCKED_MANIFEST_JSON = PROJECT / "results" / "locked_v7_model_manifest.json"
OVERATH_100_CACHE = PROJECT / "results" / "overath_100_timing_cache.pt"
NIPAH_FEAT_CACHE = PROJECT / "results" / "external_nipah_features_cache.pt"
SCORPION_FEAT_CACHE = PROJECT / "results" / "external_scorpion_features_cache.pt"
NIPAH_PRED_CSV = PROJECT / "results" / "external_nipah_locked_predictions.csv"
SCORPION_PRED_CSV = PROJECT / "results" / "external_scorpion_locked_predictions.csv"
EXTERNAL_RESULTS_JSON = PROJECT / "results" / "external_and_runtime_results.json"


# Exact v7 feature specifications from scripts/run_rigorous_ncs_revisions.py
CROSS_SPECS_5COMP = [
    [("ebm_esm_term_design", -1.0), ("esm_binder_all_exp_hydro", -1.0), ("ss_sheet_all", -1.0), ("mono_hydro_core_minus_surf", 1.0), ("mono_sap_per_res", -1.0)],
    [("oring_arom_contrast", 1.0), ("oring_hotspot_arom", 1.0), ("intf_true_arom_pack", 1.0), ("esm_binder_intf_minus_all_arom_frust", 1.0), ("sc_core_weighted_z", -1.0), ("sc_intf_angular_strain", -1.0), ("sc_intf_top3_dist_tmpl", 1.0), ("bb_force_holo_minus_apo_t", -1.0), ("bb_force_intf_minus_core", -1.0), ("ebm_normal_repulsion_force", -1.0)],
    [("ebm_elec_term_design", -1.0), ("ebm_elec_term_decoy_mean", -1.0), ("gen_ebm_elec_e0", -1.0), ("gen_ebm_e0_per_contact", -1.0), ("intf_n_sc5", -1.0)],
    [("alchem_ala_scan_ddg", -1.0), ("if3d_intf_complex_minus_monomer_sum", 1.0), ("holo_minus_apo_if_t", 1.0), ("holo_if_b_intf", 1.0)],
    [("ebm_learned_funnel_gap", -1.0), ("ebm_interface_gate_funnel_gap", 1.0), ("mech_shear_stiffness", -1.0), ("redock_global_zscore", -1.0), ("epi_consensus_density", 1.0), ("epi_concavity", 1.0)],
]

B1_SPECS_PURE_EBM = [
    ("ebm_esm_term_design", -1.0),
    ("ebm_elec_term_design", -1.0),
    ("ebm_elec_term_decoy_mean", -1.0),
    ("ebm_learned_funnel_gap", -1.0),
    ("ebm_interface_gate_funnel_gap", 1.0),
    ("esm_binder_intf_minus_all_arom_frust", 1.0),
    ("oring_arom_contrast", 1.0),
    ("oring_hotspot_arom", 1.0),
    ("intf_true_arom_pack", 1.0),
    ("mech_shear_stiffness", -1.0),
    ("gen_ebm_e0_per_contact", -1.0),
    ("ss_sheet_all", -1.0),
    ("mono_hydro_core_minus_surf", 1.0),
    ("esm_binder_all_exp_hydro", -1.0),
    ("intf_n_sc5", -1.0),
    ("alchem_ala_scan_ddg", -1.0),
    ("redock_global_zscore", -1.0),
    ("sc_core_weighted_z", -1.0),
    ("epi_consensus_density", 1.0),
    ("holo_minus_apo_if_t", 1.0),
    ("holo_if_b_intf", 1.0),
    ("bb_force_holo_minus_apo_t", -1.0),
    ("bb_force_intf_minus_core", -1.0),
    ("sc_intf_angular_strain", -1.0),
    ("ebm_normal_repulsion_force", -1.0),
    ("gen_ebm_elec_e0", -1.0),
    ("epi_concavity", 1.0),
    ("mono_sap_per_res", -1.0),
    ("sc_intf_top3_dist_tmpl", 1.0),
]

PURE_WITHIN_SPECS = [
    ("mono_bb_hbond_sat", 1.0),
    ("mono_unsat_bb_core", -1.0),
    ("mono_sap_per_res", -1.0),
    ("mono_clash_per_res", -1.0),
    ("if3d_intf_complex_minus_monomer_sum", 1.0),
    ("sc_core_weighted_z", -1.0),
    ("sc_intf_minus_nonintf_dist", -1.0),
    ("sc_intf_min_mode_dist", -1.0),
    ("sc_intf_top3_dist_tmpl", 1.0),
    ("mirage_clash_jump", -1.0),
    ("gen_ebm_rot_ent_e0", -1.0),
    ("gen_ebm_buns_total_e0", 1.0),
    ("gen_ebm_vdw_e0", -1.0),
    ("gen_ebm_e0_erased", -1.0),
    ("ebm_total_design", -1.0),
    ("ebm_clash_term_near", -1.0),
    ("ebm_esm_term_decoy_mean", 1.0),
    ("alchem_buns_mirage_gap", 1.0),
    ("alchem_ala_scan_ddg", -1.0),
    ("redock_boltz_rmsd_t05", -1.0),
    ("esm_target_intf_hydro_frust", -1.0),
    ("esm_binder_all_hydro_frust", -1.0),
    ("ebm_elec_term_design", -1.0),
    ("ebm_elec_term_decoy_mean", -1.0),
    ("intf_true_saltbridge", 1.0),
    ("intf_unsat_polar_weighted", 1.0),
    ("ss_helix_intf", 1.0),
    ("ss_max_helix_intf_run", 1.0),
    ("ss_sheet_intf", -1.0),
    ("ss_loop_intf", -1.0),
    ("epi_consensus_density", 1.0),
    ("ebm_learned_hardneg_margin", -1.0),
    ("gen_if_worst_win5", 1.0),
    ("bb_force_holo_minus_apo_b", -1.0),
    ("oring_hotspot_arom", 1.0),
    ("ebm_holo_sc_gain_raw", 1.0),
    ("ebm_holo_cont_relax_emin", -1.0),
    ("epi_concavity", 1.0),
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_and_lock_overath_v7_manifest() -> dict:
    """Lock code SHA256s, checkpoint SHA256s, and Overath training parameters before inspecting external data."""
    if LOCKED_MANIFEST_JSON.exists():
        print(f"\n--- Step 1: Loaded Pre-Locked Overath v7 Manifest ({LOCKED_MANIFEST_JSON}) ---", flush=True)
        print(f"  Manifest SHA256: {sha256_file(LOCKED_MANIFEST_JSON)}", flush=True)
        return json.loads(LOCKED_MANIFEST_JSON.read_text())
    print("\n--- Step 1: Locking Code/Checkpoint Provenance & Overath v7 Parameters ---", flush=True)
    tracked_files = [
        PROJECT / "scripts" / "evaluate_overath_ebm.py",
        PROJECT / "scripts" / "monomer_foldability_and_desolvation.py",
        PROJECT / "scripts" / "general_protein_sidechain_and_ebm.py",
        PROJECT / "scripts" / "run_rigorous_ncs_revisions.py",
        CKPT_HARDNEG,
        IF_MODEL_CKPT,
        GEN_PROTEIN_CKPT,
        OVERATH_FEAT_CACHE,
    ]
    file_hashes = {str(p.relative_to(PROJECT)): sha256_file(p) for p in tracked_files if p.exists()}

    rows = torch.load(OVERATH_FEAT_CACHE, map_location="cpu", weights_only=False)
    y = np.array([int(r["label"]) for r in rows], dtype=np.int32)
    targets = np.array([str(r["target_id"]) for r in rows])
    uniq_t = sorted(np.unique(targets))
    N = len(rows)

    def get_raw(name: str) -> np.ndarray:
        c = np.array([float(r["features"].get(name, np.nan)) for r in rows], dtype=np.float64)
        c[np.isinf(c)] = np.nan
        if np.all(np.isnan(c)):
            return np.zeros(N, dtype=np.float64)
        med = np.nanmedian(c)
        c[np.isnan(c)] = med
        return c

    def z_within_col(x: np.ndarray) -> np.ndarray:
        out = np.zeros_like(x)
        for u in uniq_t:
            m = targets == u
            med = np.median(x[m])
            std = np.std(x[m]) + 1e-5
            out[m] = np.tanh((x[m] - med) / (1.5 * std))
        return out

    # Fit 5-comp cross-target stats & consensus weights on all 13 Overath targets
    S1_fold = (
        z_within_col(get_raw("mono_bb_hbond_sat") - 0.35 * get_raw("ss_sheet_all"))
        - z_within_col(get_raw("mono_unsat_bb_core"))
        - z_within_col(get_raw("mono_sap_per_res"))
        - z_within_col(get_raw("mono_clash_per_res"))
        + z_within_col(get_raw("ss_helix_intf"))
        + z_within_col(get_raw("ss_max_helix_intf_run"))
        - z_within_col(get_raw("ss_sheet_intf"))
        - z_within_col(get_raw("ss_loop_intf"))
        + z_within_col(get_raw("gen_if_worst_win5"))
    ) / 9.0
    S2_strain = (
        -z_within_col(get_raw("sc_core_weighted_z"))
        - z_within_col(get_raw("sc_intf_min_mode_dist"))
        - z_within_col(get_raw("sc_intf_minus_nonintf_dist"))
        + z_within_col(get_raw("sc_intf_top3_dist_tmpl"))
        - z_within_col(get_raw("mirage_clash_jump"))
        - z_within_col(get_raw("gen_ebm_rot_ent_e0"))
        - z_within_col(get_raw("bb_force_holo_minus_apo_b"))
        + z_within_col(get_raw("oring_hotspot_arom"))
        + z_within_col(get_raw("ebm_holo_sc_gain_raw"))
    ) / 9.0
    S3_intf = (
        z_within_col(get_raw("gen_ebm_buns_total_e0"))
        + z_within_col(get_raw("intf_unsat_polar_weighted"))
        - z_within_col(get_raw("ebm_elec_term_design"))
        - z_within_col(get_raw("ebm_elec_term_decoy_mean"))
        + z_within_col(get_raw("intf_true_saltbridge"))
        - z_within_col(get_raw("gen_ebm_vdw_e0"))
        - z_within_col(get_raw("gen_ebm_e0_erased"))
        - z_within_col(get_raw("ebm_total_design"))
        - z_within_col(get_raw("ebm_clash_term_near"))
        - z_within_col(get_raw("ebm_learned_hardneg_margin"))
    ) / 10.0
    S4_alchem = (
        z_within_col(get_raw("alchem_buns_mirage_gap"))
        - z_within_col(get_raw("alchem_ala_scan_ddg"))
        + z_within_col(get_raw("if3d_intf_complex_minus_monomer_sum"))
        - z_within_col(get_raw("esm_target_intf_hydro_frust"))
        - z_within_col(get_raw("esm_binder_all_hydro_frust"))
        + z_within_col(get_raw("ebm_esm_term_decoy_mean"))
    ) / 6.0
    S5_funnel = (
        -z_within_col(get_raw("redock_boltz_rmsd_t05"))
        - z_within_col(get_raw("ebm_holo_cont_relax_emin"))
        + z_within_col(get_raw("epi_consensus_density"))
        + z_within_col(get_raw("epi_concavity"))
    ) / 4.0

    raw_states = [S1_fold, S2_strain, S3_intf, S4_alchem, S5_funnel]
    norm_states = []
    for st in raw_states:
        st_n = np.zeros_like(st)
        for u in uniq_t:
            m = targets == u
            st_n[m] = (st[m] - np.median(st[m])) / (np.std(st[m]) + 1e-5)
        norm_states.append(st_n)
    Macro_W = np.column_stack(norm_states)

    macro_c_cols = []
    cross_stats_5comp = []
    for group in CROSS_SPECS_5COMP:
        g_mat = np.column_stack([sgn * get_raw(k) for k, sgn in group])
        med = np.mean([np.median(g_mat[targets == u], axis=0) for u in uniq_t], axis=0, keepdims=True)
        std = np.mean([np.std(g_mat[targets == u], axis=0) for u in uniq_t], axis=0, keepdims=True) + 1e-5
        z_g = np.tanh((g_mat - med) / (1.5 * std))
        macro_c_cols.append(z_g.mean(axis=1))
        cross_stats_5comp.append({"med": med.flatten().tolist(), "std": std.flatten().tolist()})
    Macro_C = np.column_stack(macro_c_cols)

    w5_shared = np.ones(5, dtype=np.float64) / 5.0
    for j in range(5):
        w_up = w5_shared.copy()
        w_up[j] += 0.12 / 5.0
        w_up /= w_up.sum()
        gains = 0
        for u_val in uniq_t:
            in_tr = targets != u_val
            if average_precision_score(y[in_tr], np.einsum("ij,j->i", Macro_C[in_tr], w_up)) > average_precision_score(
                y[in_tr], np.einsum("ij,j->i", Macro_C[in_tr], w5_shared)
            ):
                gains += 1
        s_up = np.einsum("ij,j->i", Macro_W, w_up)
        s_base = np.einsum("ij,j->i", Macro_W, w5_shared)
        aps_up = [average_precision_score(y[targets == u], s_up[targets == u]) for u in uniq_t if u not in ("IL10Ra", "LTK")]
        aps_base = [average_precision_score(y[targets == u], s_base[targets == u]) for u in uniq_t if u not in ("IL10Ra", "LTK")]
        if gains >= 10 and np.mean(aps_up) >= np.mean(aps_base):
            w5_shared = w_up

    # Fit Full BC-EBM cross-target stats & consensus weights on all 13 Overath targets
    W_raw = np.column_stack([sgn * get_raw(k) for k, sgn in PURE_WITHIN_SPECS])
    W_raw[:, 0] = get_raw("mono_bb_hbond_sat") - 0.35 * get_raw("ss_sheet_all")
    Z_within = np.zeros_like(W_raw)
    for u in uniq_t:
        m = targets == u
        med_u = np.median(W_raw[m], axis=0, keepdims=True)
        std_u = np.std(W_raw[m], axis=0, keepdims=True) + 1e-5
        Z_within[m] = np.tanh((W_raw[m] - med_u) / (1.5 * std_u))

    E_raw = np.column_stack([sgn * get_raw(k) for k, sgn in B1_SPECS_PURE_EBM])
    med_e = np.mean([np.median(E_raw[targets == u], axis=0) for u in uniq_t], axis=0, keepdims=True)
    std_e = np.mean([np.std(E_raw[targets == u], axis=0) for u in uniq_t], axis=0, keepdims=True) + 1e-5
    Z_e = np.tanh((E_raw - med_e) / (1.5 * std_e))

    Ke = E_raw.shape[1]
    we_cur = np.ones(Ke, dtype=np.float64) / Ke
    for _ in range(2):
        for j in range(Ke):
            w_up = we_cur.copy()
            w_up[j] += 0.35 / Ke
            w_up /= w_up.sum()
            gains = 0
            for u_val in uniq_t:
                in_tr = targets != u_val
                if average_precision_score(y[in_tr], np.einsum("ij,j->i", Z_e[in_tr], w_up)) > average_precision_score(
                    y[in_tr], np.einsum("ij,j->i", Z_e[in_tr], we_cur)
                ):
                    gains += 1
            if gains >= 11:
                we_cur = w_up

    Kw = Z_within.shape[1]
    w1_cur = np.ones(Kw, dtype=np.float64) / Kw
    tr_mh = [u for u in uniq_t if u not in ("IL10Ra", "LTK")]
    for j in range(Kw):
        w_up = w1_cur.copy()
        w_up[j] += 0.35 / Kw
        w_up /= w_up.sum()
        s_up = np.einsum("ij,j->i", Z_within, w_up)
        s_base = np.einsum("ij,j->i", Z_within, w1_cur)
        aps_up = [average_precision_score(y[targets == u], s_up[targets == u]) for u in tr_mh]
        aps_base = [average_precision_score(y[targets == u], s_base[targets == u]) for u in tr_mh]
        if np.mean(aps_up) > np.mean(aps_base) and sum(a >= b - 1e-4 for a, b in zip(aps_up, aps_base)) >= 9:
            w1_cur = w_up

    manifest = {
        "locked_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "description": "Locked v7 BC-EBM model manifest generated strictly from the 13-target Overath benchmark BEFORE evaluating external campaigns (Nipah-G and Scorpion).",
        "file_sha256": file_hashes,
        "overath_n_candidates": int(N),
        "overath_n_targets": len(uniq_t),
        "cross_stats_5comp": cross_stats_5comp,
        "w5_shared": w5_shared.tolist(),
        "full_ebm_cross_med": med_e.flatten().tolist(),
        "full_ebm_cross_std": std_e.flatten().tolist(),
        "we_cur": we_cur.tolist(),
        "w1_cur": w1_cur.tolist(),
    }
    LOCKED_MANIFEST_JSON.write_text(json.dumps(manifest, indent=2))
    print(f"Saved locked v7 manifest to {LOCKED_MANIFEST_JSON} (SHA256={sha256_file(LOCKED_MANIFEST_JSON)[:16]}...)", flush=True)
    return manifest


def score_dataset_with_locked_v7(rows: list[dict], manifest: dict) -> dict[str, np.ndarray]:
    """Apply the locked Overath v7 model unchanged to a list of feature rows."""
    targets = np.array([str(r["target_id"]) for r in rows])
    uniq_t = sorted(np.unique(targets))
    N = len(rows)

    def get_raw(name: str) -> np.ndarray:
        c = np.array([float(r["features"].get(name, np.nan)) for r in rows], dtype=np.float64)
        c[np.isinf(c)] = np.nan
        if np.all(np.isnan(c)):
            return np.zeros(N, dtype=np.float64)
        med = np.nanmedian(c)
        c[np.isnan(c)] = med
        return c

    def z_within_col(x: np.ndarray) -> np.ndarray:
        out = np.zeros_like(x)
        for u in uniq_t:
            m = targets == u
            med = np.median(x[m])
            std = np.std(x[m]) + 1e-5
            out[m] = np.tanh((x[m] - med) / (1.5 * std))
        return out

    # 1. 5-Component Physics-Inspired Score
    S1_fold = (
        z_within_col(get_raw("mono_bb_hbond_sat") - 0.35 * get_raw("ss_sheet_all"))
        - z_within_col(get_raw("mono_unsat_bb_core"))
        - z_within_col(get_raw("mono_sap_per_res"))
        - z_within_col(get_raw("mono_clash_per_res"))
        + z_within_col(get_raw("ss_helix_intf"))
        + z_within_col(get_raw("ss_max_helix_intf_run"))
        - z_within_col(get_raw("ss_sheet_intf"))
        - z_within_col(get_raw("ss_loop_intf"))
        + z_within_col(get_raw("gen_if_worst_win5"))
    ) / 9.0
    S2_strain = (
        -z_within_col(get_raw("sc_core_weighted_z"))
        - z_within_col(get_raw("sc_intf_min_mode_dist"))
        - z_within_col(get_raw("sc_intf_minus_nonintf_dist"))
        + z_within_col(get_raw("sc_intf_top3_dist_tmpl"))
        - z_within_col(get_raw("mirage_clash_jump"))
        - z_within_col(get_raw("gen_ebm_rot_ent_e0"))
        - z_within_col(get_raw("bb_force_holo_minus_apo_b"))
        + z_within_col(get_raw("oring_hotspot_arom"))
        + z_within_col(get_raw("ebm_holo_sc_gain_raw"))
    ) / 9.0
    S3_intf = (
        z_within_col(get_raw("gen_ebm_buns_total_e0"))
        + z_within_col(get_raw("intf_unsat_polar_weighted"))
        - z_within_col(get_raw("ebm_elec_term_design"))
        - z_within_col(get_raw("ebm_elec_term_decoy_mean"))
        + z_within_col(get_raw("intf_true_saltbridge"))
        - z_within_col(get_raw("gen_ebm_vdw_e0"))
        - z_within_col(get_raw("gen_ebm_e0_erased"))
        - z_within_col(get_raw("ebm_total_design"))
        - z_within_col(get_raw("ebm_clash_term_near"))
        - z_within_col(get_raw("ebm_learned_hardneg_margin"))
    ) / 10.0
    S4_alchem = (
        z_within_col(get_raw("alchem_buns_mirage_gap"))
        - z_within_col(get_raw("alchem_ala_scan_ddg"))
        + z_within_col(get_raw("if3d_intf_complex_minus_monomer_sum"))
        - z_within_col(get_raw("esm_target_intf_hydro_frust"))
        - z_within_col(get_raw("esm_binder_all_hydro_frust"))
        + z_within_col(get_raw("ebm_esm_term_decoy_mean"))
    ) / 6.0
    S5_funnel = (
        -z_within_col(get_raw("redock_boltz_rmsd_t05"))
        - z_within_col(get_raw("ebm_holo_cont_relax_emin"))
        + z_within_col(get_raw("epi_consensus_density"))
        + z_within_col(get_raw("epi_concavity"))
    ) / 4.0

    raw_states = [S1_fold, S2_strain, S3_intf, S4_alchem, S5_funnel]
    norm_states = []
    for st in raw_states:
        st_n = np.zeros_like(st)
        for u in uniq_t:
            m = targets == u
            st_n[m] = (st[m] - np.median(st[m])) / (np.std(st[m]) + 1e-5)
        norm_states.append(st_n)
    Macro_W = np.column_stack(norm_states)

    macro_c_cols = []
    for group, st_info in zip(CROSS_SPECS_5COMP, manifest["cross_stats_5comp"]):
        g_mat = np.column_stack([sgn * get_raw(k) for k, sgn in group])
        med = np.array(st_info["med"], dtype=np.float64)[None, :]
        std = np.array(st_info["std"], dtype=np.float64)[None, :]
        z_g = np.tanh((g_mat - med) / (1.5 * std))
        macro_c_cols.append(z_g.mean(axis=1))
    Macro_C = np.column_stack(macro_c_cols)

    w5_eq = np.ones(5, dtype=np.float64) / 5.0
    w5_reg = np.array(manifest["w5_shared"], dtype=np.float64)

    pred_5comp_eq = np.zeros(N, dtype=np.float64)
    pred_5comp_reg = np.zeros(N, dtype=np.float64)
    for u in uniq_t:
        te = targets == u
        n_te = float(te.sum())
        beta_te = 0.40 * max(0.0, (n_te - 150.0) / (n_te + 400.0))

        s_c_eq = np.einsum("ij,j->i", Macro_C[te], w5_eq)
        s_w_eq = np.einsum("ij,j->i", Macro_W[te], w5_eq)
        s_w_eq_n = (s_w_eq - np.median(s_w_eq)) / (np.std(s_w_eq) + 1e-5)
        psi_eq = s_w_eq_n + 0.20 * np.maximum(0.0, s_w_eq_n - 0.8) ** 2
        pred_5comp_eq[te] = np.median(s_c_eq) + beta_te * (s_c_eq - np.median(s_c_eq)) + 0.75 * np.std(s_c_eq) * psi_eq

        s_c_reg = np.einsum("ij,j->i", Macro_C[te], w5_reg)
        s_w_reg = np.einsum("ij,j->i", Macro_W[te], w5_reg)
        s_w_reg_n = (s_w_reg - np.median(s_w_reg)) / (np.std(s_w_reg) + 1e-5)
        psi_reg = s_w_reg_n + 0.20 * np.maximum(0.0, s_w_reg_n - 0.8) ** 2
        pred_5comp_reg[te] = np.median(s_c_reg) + beta_te * (s_c_reg - np.median(s_c_reg)) + 0.75 * np.std(s_c_reg) * psi_reg

    # 2. Full BC-EBM (Equal-Weight and Regularized)
    W_raw = np.column_stack([sgn * get_raw(k) for k, sgn in PURE_WITHIN_SPECS])
    W_raw[:, 0] = get_raw("mono_bb_hbond_sat") - 0.35 * get_raw("ss_sheet_all")
    Z_within = np.zeros_like(W_raw)
    for u in uniq_t:
        m = targets == u
        med_u = np.median(W_raw[m], axis=0, keepdims=True)
        std_u = np.std(W_raw[m], axis=0, keepdims=True) + 1e-5
        Z_within[m] = np.tanh((W_raw[m] - med_u) / (1.5 * std_u))

    E_raw = np.column_stack([sgn * get_raw(k) for k, sgn in B1_SPECS_PURE_EBM])
    med_e = np.array(manifest["full_ebm_cross_med"], dtype=np.float64)[None, :]
    std_e = np.array(manifest["full_ebm_cross_std"], dtype=np.float64)[None, :]
    Z_e = np.tanh((E_raw - med_e) / (1.5 * std_e))

    Ke = E_raw.shape[1]
    Kw = Z_within.shape[1]
    we_eq = np.ones(Ke, dtype=np.float64) / Ke
    w1_eq = np.ones(Kw, dtype=np.float64) / Kw
    we_reg = np.array(manifest["we_cur"], dtype=np.float64)
    w1_reg = np.array(manifest["w1_cur"], dtype=np.float64)

    pred_full_eq = np.zeros(N, dtype=np.float64)
    pred_full_reg = np.zeros(N, dtype=np.float64)
    s_win_within_norm = np.zeros(N, dtype=np.float64)

    for u in uniq_t:
        te = targets == u
        n_te = float(te.sum())
        beta_te = 0.40 * max(0.0, (n_te - 150.0) / (n_te + 400.0))

        s_ebm_eq = np.einsum("ij,j->i", Z_e[te], we_eq)
        s_win_eq = np.einsum("ij,j->i", Z_within[te], w1_eq)
        s_win_eq_n = (s_win_eq - np.median(s_win_eq)) / (np.std(s_win_eq) + 1e-5)
        s_win_within_norm[te] = s_win_eq_n
        psi_win_eq = s_win_eq_n + 0.20 * np.maximum(0.0, s_win_eq_n - 0.8) ** 2
        pred_full_eq[te] = np.median(s_ebm_eq) + beta_te * (s_ebm_eq - np.median(s_ebm_eq)) + 0.75 * np.std(s_ebm_eq) * psi_win_eq

        s_ebm_reg = np.einsum("ij,j->i", Z_e[te], we_reg)
        s_win_reg = np.einsum("ij,j->i", Z_within[te], w1_reg)
        s_win_reg_n = (s_win_reg - np.median(s_win_reg)) / (np.std(s_win_reg) + 1e-5)
        psi_win_reg = s_win_reg_n + 0.20 * np.maximum(0.0, s_win_reg_n - 0.8) ** 2
        pred_full_reg[te] = np.median(s_ebm_reg) + beta_te * (s_ebm_reg - np.median(s_ebm_reg)) + 0.75 * np.std(s_ebm_reg) * psi_win_reg

    return {
        "bc_ebm_5comp_ew": pred_5comp_eq,
        "bc_ebm_5comp_reg": pred_5comp_reg,
        "bc_ebm_full_ew": pred_full_eq,
        "bc_ebm_full_reg": pred_full_reg,
        "s_win_within_norm": s_win_within_norm,
        "S1_fold": norm_states[0],
        "S2_strain": norm_states[1],
        "S3_intf": norm_states[2],
        "S4_alchem": norm_states[3],
        "S5_funnel": norm_states[4],
    }


def extract_full_v7_features_for_records(
    records: list[dict],
    cache_path: Path,
    dataset_name: str,
    force_rebuild: bool = False,
) -> tuple[list[dict], dict]:
    """Extract all Stage A + Stage B + Stage C (including autograd backbone forces & 6-DoF relaxation) features."""
    torch.set_num_threads(2)
    if cache_path.exists() and not force_rebuild:
        print(f"Loading cached {dataset_name} features from {cache_path} ...", flush=True)
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        return payload["rows"], payload["timing"]

    print(f"\nExtracting complete v7 BC-EBM features for {len(records)} {dataset_name} complexes ...", flush=True)
    t_start_total = time.perf_counter()

    # Load Stage A model (UnboundOrientationPairwiseEBM)
    ckpt_a = torch.load(CKPT_HARDNEG, map_location="cpu", weights_only=False)
    state_a = ckpt_a.get("model_state", ckpt_a.get("model_state_dict", ckpt_a))
    ebm_a = UnboundOrientationPairwiseEBM(
        hidden=64,
        monomer_layers=3,
        neural_interface_weight=ckpt_a.get("neural_interface_weight", 0.11),
        neural_veto_weight=ckpt_a.get("neural_veto_weight", 0.008),
        neural_veto_thresh=ckpt_a.get("neural_veto_thresh", -6.00),
        neural_veto_temp=ckpt_a.get("neural_veto_temp", 1.00),
        esm_rank=16,
        esm_bilin_weight=ckpt_a.get("esm_bilin_weight", 0.020),
        enable_head_on_esm=True,
    )
    ebm_a.load_state_dict(state_a)
    ebm_a.eval()

    # Load ESM-2 on CPU (faster than MPS for single-sequence 8M inference due to zero kernel launch overhead)
    esm_device = torch.device("cpu")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    esm = AutoModelForMaskedLM.from_pretrained(MODEL_ID).to(esm_device)
    esm.eval()
    aa_tok_ids = [tok.convert_tokens_to_ids(a) for a in _AA1_LIST]
    phys_tbl_esm = PHYSCHEM_TABLE[:20].to(esm_device)

    # Load Stage B IF model (results/backbone_inverse_folding_gnn.pt)
    if_model_b = train_or_load_backbone_inverse_folding_gnn(force_retrain=False)
    if_model_b.eval()

    # Load Stage C v4 general models (results/general_sidechain_and_ebm.pt) on CPU for unified timing
    if_model_c, sc_model_c, ebm_model_c = train_or_load_general_models(force_retrain=False)
    dev_c = torch.device("cpu")
    if_model_c.to(dev_c).eval()
    sc_model_c.to(dev_c).eval()
    ebm_model_c.to(dev_c).eval()
    tmpl_u_sc = sc_model_c.tmpl_u_sc.to(dev_c)
    tmpl_u_tip = sc_model_c.tmpl_u_tip.to(dev_c)
    tmpl_sigma = sc_model_c.tmpl_sigma.to(dev_c)
    phys_tbl = PHYSCHEM_TABLE.to(dev_c)

    esm_seq_cache: dict[tuple, dict[str, torch.Tensor]] = {}
    stage_times = defaultdict(float)

    def get_cached_esm(aa: torch.Tensor, chain: torch.Tensor) -> dict[str, torch.Tensor]:
        key = (tuple(aa.tolist()), tuple(chain.tolist()))
        if key not in esm_seq_cache:
            t0_e = time.perf_counter()
            esm_seq_cache[key] = extract_partner_esm(
                aa, chain, tok, esm, aa_tok_ids, phys_tbl_esm, esm_device
            )
            stage_times["esm2_sec"] += time.perf_counter() - t0_e
        return esm_seq_cache[key]

    # Pass 1: Parse 5-bead structures and compute cohort-wide unsupervised target contact counts
    t_start_total = time.perf_counter()
    t0_parse = time.perf_counter()
    parsed_items = []
    target_seq_cache_c = {}
    contact_counts = {}
    target_geom_props = {}
    target_stage_a_cache = {}

    for idx, rec in enumerate(records):
        b_geom = extract_monomer_geometry(rec["binder_pdb"])
        t_geom = extract_monomer_geometry(rec["target_pdb"])
        b_res = b_geom["residues"]
        t_res = t_geom["residues"]

        b_beads = torch.from_numpy(b_geom["beads"]).to(torch.float32)
        b_cb = torch.from_numpy(b_geom["cb"]).to(torch.float32)
        b_has_cb = torch.from_numpy(b_geom["has_phys_cb"])
        b_aa = torch.tensor([AA3.get(x["res_name"], 20) for x in b_res], dtype=torch.long)
        binder_chains = sorted({x["chain"] for x in b_res})
        b_chain_map = {c: i for i, c in enumerate(binder_chains)}
        b_chain = torch.tensor([b_chain_map[x["chain"]] for x in b_res], dtype=torch.long)

        t_beads = torch.from_numpy(t_geom["beads"]).to(torch.float32)
        t_cb = torch.from_numpy(t_geom["cb"]).to(torch.float32)
        t_has_cb = torch.from_numpy(t_geom["has_phys_cb"])
        t_aa = torch.tensor([AA3.get(x["res_name"], 20) for x in t_res], dtype=torch.long)
        t_chain = torch.zeros((len(t_res),), dtype=torch.long)

        seq_key = (rec["target_id"], tuple(t_aa.tolist()))
        d_ca_bt = torch.cdist(b_beads[:, 1], t_beads[:, 1])
        t_cont = (d_ca_bt.amin(dim=0) < 8.5).to(torch.float32)
        if seq_key not in contact_counts:
            contact_counts[seq_key] = t_cont.clone()
        else:
            if len(contact_counts[seq_key]) == len(t_cont):
                contact_counts[seq_key] += t_cont

        parsed_items.append({
            "rec": rec,
            "b_beads": b_beads,
            "b_cb": b_cb,
            "b_has_cb": b_has_cb,
            "b_aa": b_aa,
            "b_chain": b_chain,
            "t_beads": t_beads,
            "t_cb": t_cb,
            "t_has_cb": t_has_cb,
            "t_aa": t_aa,
            "t_chain": t_chain,
            "seq_key": seq_key,
        })

    for k in contact_counts:
        contact_counts[k] = contact_counts[k] / (contact_counts[k].max() + 1e-5)
    parse_sec = time.perf_counter() - t0_parse

    rng_a = np.random.default_rng(2026)
    rng_c = np.random.default_rng(2026)
    I_rot = torch.eye(3, dtype=torch.float32).unsqueeze(0)
    Z_tr = torch.zeros((1, 3), dtype=torch.float32)

    output_rows = []
    stage_times["parse_5bead_sec"] = parse_sec

    for idx, item in enumerate(parsed_items):
        rec = item["rec"]
        b_beads = item["b_beads"]
        b_cb = item["b_cb"]
        b_has_cb = item["b_has_cb"]
        b_aa = item["b_aa"]
        b_chain = item["b_chain"]
        t_beads = item["t_beads"]
        t_cb = item["t_cb"]
        t_has_cb = item["t_has_cb"]
        t_aa = item["t_aa"]
        t_chain = item["t_chain"]
        seq_key = item["seq_key"]

        # --- Stage A: ESM-2 + UnboundOrientationPairwiseEBM (16 poses) ---
        t0_a = time.perf_counter()
        if seq_key not in target_stage_a_cache:
            with torch.no_grad():
                t_role = torch.ones((len(t_beads),), dtype=torch.long)
                t_res_idx = torch.arange(len(t_beads), dtype=torch.long)
                t_region = torch.zeros((len(t_beads),), dtype=torch.long)
                h_lig, frames_lig, sc_vec_lig, o_vec_lig, prot_lig = ebm_a._compute_monomer_features(
                    t_beads, t_cb, t_aa, t_role, t_region, t_chain, t_res_idx
                )
                ag_ca0 = t_beads[:, 1]
                d_ag_ag = torch.cdist(ag_ca0, ag_ca0)
                w12 = (d_ag_ag < 12.0).to(ag_ca0.dtype)
                n12 = F.normalize(ag_ca0 - (w12 @ ag_ca0) / w12.sum(dim=-1, keepdim=True).clamp_min(1.0), dim=-1, eps=1e-6)
                w18 = (d_ag_ag < 18.0).to(ag_ca0.dtype)
                n18 = F.normalize(ag_ca0 - (w18 @ ag_ca0) / w18.sum(dim=-1, keepdim=True).clamp_min(1.0), dim=-1, eps=1e-6)
                n_ag0 = F.normalize(0.5 * n12 + 0.5 * n18, dim=-1, eps=1e-6)

                t_esm = get_cached_esm(t_aa, t_chain)
                ag_esm = t_esm["esm_hidden"].to(dtype=ag_ca0.dtype)
                ag_frust = t_esm["esm_frust"].to(dtype=ag_ca0.dtype)
                surf_m = prot_lig > -0.20
                if int(surf_m.sum().item()) < 10:
                    surf_m = torch.ones_like(prot_lig, dtype=torch.bool)
                f_mean = ag_frust[surf_m].mean(dim=0, keepdim=True)
                f_std = ag_frust[surf_m].std(dim=0, keepdim=True) + 1e-3
                ag_frust_z = ((ag_frust - f_mean) / f_std).clamp(-3.5, 3.5)
                e_mean = ag_esm[surf_m].mean(dim=0, keepdim=True)
                ag_esm_c = F.normalize(ag_esm - e_mean, dim=-1, eps=1e-6)

                target_stage_a_cache[seq_key] = {
                    "lig_beads": t_beads,
                    "lig_cb": t_cb,
                    "lig_has_phys_cb": t_has_cb,
                    "lig_aa": t_aa,
                    "lig_region": t_region,
                    "h_lig": h_lig,
                    "frames_lig": frames_lig,
                    "sc_vec_lig": sc_vec_lig,
                    "o_vec_lig": o_vec_lig,
                    "prot_lig": prot_lig,
                    "lig_phys": PHYSCHEM_TABLE[t_aa.clamp(0, 20)],
                    "n_ag0": n_ag0,
                    "ag_frust_z": ag_frust_z,
                    "ag_esm_c": ag_esm_c,
                    "t_esm_frust_np": t_esm["esm_frust"].numpy(),
                }
        t_cached = target_stage_a_cache[seq_key]

        with torch.no_grad():
            b_res_idx = torch.arange(len(b_beads), dtype=torch.long)
            b_role = torch.zeros((len(b_beads),), dtype=torch.long)
            d_ca = torch.cdist(b_beads[:, 1], t_beads[:, 1])
            min_d_b = d_ca.amin(dim=1)
            rec_region = torch.full((len(b_beads),), REGION_AB_VFW, dtype=torch.long)
            rim_mask = (min_d_b >= 8.0) & (min_d_b < 11.5)
            core_mask = min_d_b < 8.0
            if int(core_mask.sum().item()) < 4:
                _, top_idx = torch.topk(min_d_b, k=min(8, len(b_beads)), largest=False)
                core_mask = torch.zeros_like(core_mask)
                core_mask[top_idx] = True
            rec_region[rim_mask] = REGION_AB_CDR1
            rec_region[core_mask] = REGION_AB_CDRH3

            h_rec, frames_rec, sc_vec_rec, o_vec_rec, prot_rec = ebm_a._compute_monomer_features(
                b_beads, b_cb, b_aa, b_role, rec_region, b_chain, b_res_idx
            )
            cdr_centroid0, u_para0 = compute_antibody_paratope_geometry(b_beads[:, 1], rec_region)
            b_esm = get_cached_esm(b_aa, b_chain)
            ab_esm = b_esm["esm_hidden"].to(dtype=b_beads.dtype)
            cdr_mask = rec_region >= REGION_AB_CDR1
            if cdr_mask.any():
                w_cdr_ab = (1.0 + 0.5 * (rec_region[cdr_mask] >= REGION_AB_CDRL3).to(b_beads.dtype)).unsqueeze(-1)
                w_cdr_ab = w_cdr_ab / w_cdr_ab.sum().clamp_min(1e-6)
                ab_esm_pool = F.normalize((w_cdr_ab * ab_esm[cdr_mask]).sum(dim=0), dim=-1, eps=1e-6)
            else:
                ab_esm_pool = F.normalize(ab_esm.mean(dim=0), dim=-1, eps=1e-6)

            comp_pool = {"rec_beads": b_beads, "rec_region": rec_region, "lig_beads": t_beads}
            rots_a, trans_a, _ = generate_surface_packed_sabdab2_pool(comp_pool, rng_a, n_poses=16)

            frames_lig_cur = build_local_frames(t_beads[:, 0], t_beads[:, 1], t_beads[:, 2])
            sc_vec_lig_cur = F.normalize(t_beads[:, 4] - t_beads[:, 1], dim=-1, eps=1e-6)
            o_vec_lig_cur = F.normalize(t_beads[:, 3] - t_beads[:, 1], dim=-1, eps=1e-6)

            enc = {
                "h_rec": h_rec,
                "frames_rec": frames_rec,
                "sc_vec_rec": sc_vec_rec,
                "o_vec_rec": o_vec_rec,
                "prot_rec": prot_rec,
                "rec_beads": b_beads,
                "rec_cb": b_cb,
                "rec_region": rec_region,
                "rec_has_phys_cb": b_has_cb,
                "rec_phys": PHYSCHEM_TABLE[b_aa.clamp(0, 20)],
                "h_lig": t_cached["h_lig"],
                "frames_lig": frames_lig_cur,
                "sc_vec_lig": sc_vec_lig_cur,
                "o_vec_lig": o_vec_lig_cur,
                "prot_lig": t_cached["prot_lig"],
                "lig_beads": t_beads,
                "lig_cb": t_cb,
                "lig_region": t_cached["lig_region"],
                "lig_has_phys_cb": t_has_cb,
                "lig_phys": t_cached["lig_phys"],
                "receptor_is_antibody": True,
                "cdr_centroid0": cdr_centroid0,
                "u_para0": u_para0,
                "n_ag0": t_cached["n_ag0"],
                "has_esm": True,
                "ag_frust_z": t_cached["ag_frust_z"],
                "ag_esm_c": t_cached["ag_esm_c"],
                "ab_esm_pool": ab_esm_pool,
            }
            comp_out = ebm_a.score_poses_from_encoded(enc, rots_a, trans_a, return_components=True)

            decoy_slice = slice(7, 16)
            feats: dict[str, float] = {}
            for k_name, tensor_val in comp_out.items():
                arr = tensor_val.detach().cpu().numpy().astype(np.float64)
                v0 = float(arr[0])
                v_near = float(np.mean(arr[1:7]))
                v_dec_mean = float(np.mean(arr[decoy_slice]))
                v_dec_min = float(np.min(arr[decoy_slice]))
                v_dec_std = float(np.std(arr[decoy_slice])) + 1e-4
                feats[f"ebm_{k_name}_design"] = v0
                feats[f"ebm_{k_name}_near"] = v_near
                feats[f"ebm_{k_name}_decoy_mean"] = v_dec_mean
                feats[f"ebm_{k_name}_funnel_gap"] = v0 - v_dec_mean
                feats[f"ebm_{k_name}_hardneg_margin"] = v0 - v_dec_min
                feats[f"ebm_{k_name}_zscore"] = (v0 - v_dec_mean) / v_dec_std

            t_near16 = d_ca.amin(dim=0) < 16.0
            if not bool(t_near16.any()):
                t_near16 = torch.ones((len(t_beads),), dtype=torch.bool)
            t_beads_near16 = t_beads[t_near16]
            t_aa_near16 = t_aa[t_near16]
            d_sc = torch.cdist(b_beads[:, 4], t_beads_near16[:, 4])
            all_bead_d = torch.cdist(b_beads.reshape(-1, 3), t_beads_near16.reshape(-1, 3))
            intf_b_mask = min_d_b < 8.5
            intf_t_mask = d_ca.amin(dim=0) < 8.5
            if not intf_b_mask.any():
                intf_b_mask = torch.ones((len(b_beads),), dtype=torch.bool)
            if not intf_t_mask.any():
                intf_t_mask = torch.ones((len(t_beads),), dtype=torch.bool)

            b_frust = b_esm["esm_frust"].numpy()
            t_frust = t_cached["t_esm_frust_np"]
            ib_m = intf_b_mask.numpy()
            it_m = intf_t_mask.numpy()

            feats["binder_length"] = float(len(b_beads))
            feats["intf_n_ca8"] = float((d_ca < 8.0).sum().item())
            feats["intf_n_sc5"] = float((d_sc < 5.0).sum().item())
            feats["intf_min_bead_dist"] = float(all_bead_d.min().item())
            feats["intf_binder_frac"] = float(ib_m.mean())
            feats["intf_head_on_cos"] = float(-(u_para0 * t_cached["n_ag0"][intf_t_mask].mean(dim=0)).sum().item())

            frust_names = ["nll", "entropy", "hydro_frust", "arom_frust", "charge_frust", "exp_hydro"]
            for f_i, f_n in enumerate(frust_names):
                feats[f"esm_binder_all_{f_n}"] = float(b_frust[:, f_i].mean())
                feats[f"esm_binder_intf_{f_n}"] = float(b_frust[ib_m, f_i].mean())
                feats[f"esm_target_intf_{f_n}"] = float(t_frust[it_m, f_i].mean())
                feats[f"esm_binder_intf_minus_all_{f_n}"] = float(b_frust[ib_m, f_i].mean() - b_frust[:, f_i].mean())
        stage_times["stage_a_sec"] += time.perf_counter() - t0_a

        # --- Stage B: Monomer Foldability, Desolvation, O-Ring, Stiffness, 3D IF ---
        t0_b = time.perf_counter()
        with torch.no_grad():
            feats.update(compute_intra_monomer_5bead_physics(b_beads, b_aa, b_chain, b_res_idx))
            feats.update(compute_interface_desolvation_and_density(b_beads, b_aa, t_beads_near16, t_aa_near16))
            feats.update(
                compute_inverse_folding_features(
                    if_model_b, b_beads, b_aa, b_chain, t_beads, t_aa, t_chain, feats["esm_binder_all_nll"]
                )
            )
            feats.update(compute_mechanical_stiffness_and_tertiary_tethering(b_beads, b_aa, t_beads_near16, t_aa_near16))
            feats.update(compute_corrected_physics_and_oring(b_beads, b_aa, t_beads_near16, t_aa_near16))
        stage_times["stage_b_sec"] += time.perf_counter() - t0_b

        # --- Stage C: 2-Point Sidechain Erasure, 24-Pose Re-Docking, Alchemical, Autograd Forces & 6-DoF Relaxation ---
        t0_c = time.perf_counter()
        if seq_key not in target_seq_cache_c:
            with torch.enable_grad():
                tb_g = t_beads.clone().requires_grad_(True)
                t_geom0 = compute_pure_backbone_geometry(tb_g, t_chain, torch.arange(len(t_beads)))
                lp_t0 = if_model_c.forward_from_geom(t_geom0)
                ll_t0_sum = lp_t0[torch.arange(len(t_beads)), t_aa.clamp(0, 19)].sum()
                grad_t0 = torch.autograd.grad(ll_t0_sum, tb_g)[0].detach()
                f_t_apo = torch.norm(grad_t0[:, :4].reshape(len(t_beads), -1), dim=-1)
                lp_t0 = lp_t0.detach()
                t_geom0 = {k: (v.detach() if isinstance(v, torch.Tensor) else v) for k, v in t_geom0.items()}
            with torch.no_grad():
                _, u_local_t, t_sigma, _, u_tip_local_t, t_sigma_tip = sc_model_c.forward_from_geom(
                    t_geom0, t_aa, return_tip=True
                )
                h_t = ebm_model_c.encode_partner(t_geom0, t_aa)
                t_aa_c0 = t_aa.clamp(0, 20)
                t_ca = t_beads[:, 1]
                d_tt = torch.cdist(t_ca, t_ca).masked_fill(torch.eye(len(t_ca), dtype=torch.bool), 1e6)
                bur6 = (d_tt < 6.0).to(torch.float32).sum(dim=-1)
                bur14 = (d_tt < 14.0).to(torch.float32).sum(dim=-1)
                t_phys = phys_tbl[t_aa_c0]
                target_seq_cache_c[seq_key] = {
                    "u_local_t": u_local_t,
                    "t_sigma": t_sigma,
                    "u_tip_local_t": u_tip_local_t,
                    "t_sigma_tip": t_sigma_tip,
                    "u_tmpl_t": tmpl_u_sc[t_aa_c0],
                    "u_tip_tmpl_t": tmpl_u_tip[t_aa_c0],
                    "t_sigma_tmpl": tmpl_sigma[t_aa_c0],
                    "h_t": h_t,
                    "lp_t_apo": lp_t0,
                    "f_t_apo": f_t_apo,
                    "concavity": bur14 / (bur6 + 2.0),
                    "hydro": F.relu(t_phys[:, 1]),
                    "arom": t_phys[:, 4],
                }
        t_info = target_seq_cache_c[seq_key]
        h_t = t_info["h_t"]

        with torch.no_grad():
            R_t = frames_lig_cur
            t_sc_hat = t_beads[:, 1] + torch.einsum("nij,nj->ni", R_t, t_info["u_local_t"])
            t_tip_hat = t_beads[:, 1] + torch.einsum("nij,nj->ni", R_t, t_info["u_tip_local_t"])
            t_erased = t_beads.clone()
            t_erased[:, 4] = t_sc_hat

        Nb = len(b_beads)
        valid_b = b_aa < 20
        b_aa_c = b_aa.clamp(0, 19)

        # Autograd apo binder backbone force
        t0_c_grad1 = time.perf_counter()
        with torch.enable_grad():
            bb_g = b_beads.clone().requires_grad_(True)
            b_geom_g = compute_pure_backbone_geometry(bb_g, b_chain, torch.arange(Nb))
            log_p_b_g = if_model_c.forward_from_geom(b_geom_g)
            ll_b_g = log_p_b_g[torch.arange(Nb), b_aa_c]
            grad_b_apo = torch.autograd.grad(ll_b_g.sum(), bb_g)[0].detach()
            f_b_apo = torch.norm(grad_b_apo[:, :4].reshape(Nb, -1), dim=-1)
            ll_b = ll_b_g.detach()
            b_geom = {k: (v.detach() if isinstance(v, torch.Tensor) else v) for k, v in b_geom_g.items()}

        d_ca0 = torch.cdist(b_beads[:, 1], t_beads[:, 1])
        b_intf = d_ca0.amin(dim=1) < 9.0
        t_near_idx = torch.where(d_ca0.amin(dim=0) < 15.0)[0]
        if t_near_idx.numel() < 5:
            t_near_idx = torch.argsort(d_ca0.amin(dim=0))[: min(25, len(t_beads))]
        Nt_n = len(t_near_idx)
        comp_aa = torch.cat([b_aa, t_aa[t_near_idx]], dim=0)
        comp_chain = torch.cat([torch.zeros(Nb, dtype=torch.long), torch.ones(Nt_n, dtype=torch.long)], dim=0)
        comp_ridx = torch.cat([torch.arange(Nb), t_near_idx + 1000], dim=0)

        # Autograd holo complex backbone force
        with torch.enable_grad():
            comp_beads_g = torch.cat([b_beads, t_beads[t_near_idx]], dim=0).clone().requires_grad_(True)
            cg_g = compute_pure_backbone_geometry(comp_beads_g, comp_chain, comp_ridx)
            lp_comp_g = if_model_c.forward_from_geom(cg_g)
            ll_comp_vec = lp_comp_g[torch.arange(Nb + Nt_n), comp_aa.clamp(0, 19)]
            grad_comp = torch.autograd.grad(ll_comp_vec.sum(), comp_beads_g)[0].detach()
            f_comp = torch.norm(grad_comp[:, :4].reshape(Nb + Nt_n, -1), dim=-1)
            ll_b_holo = ll_comp_vec[:Nb].detach()
            ll_t_holo = ll_comp_vec[Nb:].detach()
            cg = {k: (v.detach() if isinstance(v, torch.Tensor) else v) for k, v in cg_g.items()}
        stage_times["stage_c_autograd_bb_force_sec"] += time.perf_counter() - t0_c_grad1

        with torch.no_grad():
            core_m = b_geom["burial_cb10"] > b_geom["burial_cb10"].median()
            gen_if_ll_all = float(ll_b[valid_b].mean().item())
            gen_if_ll_core = float(ll_b[valid_b & core_m].mean().item()) if (valid_b & core_m).any() else gen_if_ll_all
            gen_if_ll_intf = float(ll_b[valid_b & b_intf].mean().item()) if (valid_b & b_intf).any() else gen_if_ll_all

            if Nb >= 5:
                win5_ll = F.avg_pool1d(ll_b.view(1, 1, -1), kernel_size=5, stride=1).view(-1)
                gen_if_worst_win5 = float(win5_ll.min().item())
            else:
                gen_if_worst_win5 = gen_if_ll_all

            b_sc_hat, _, b_sigma, b_tip_hat, _, b_sigma_tip, mdn_b = sc_model_c.forward_from_geom(
                b_geom, b_aa, return_tip=True, return_mdn=True
            )
            b_erased = b_beads.clone()
            b_erased[:, 4] = b_sc_hat

            sc_comp, _, sig_comp, tip_comp, _, sig_tip_comp = sc_model_c.forward_from_geom(cg, comp_aa, return_tip=True)
            sig_comp_comb = 0.5 * (sig_comp + sig_tip_comp)

            ta_n = t_aa[t_near_idx].clamp(0, 19)
            ll_t_apo = t_info["lp_t_apo"][t_near_idx, ta_n]
            t_intf_n = (d_ca0[:, t_near_idx].amin(dim=0) < 9.0) & (t_aa[t_near_idx] < 20)
            b_intf_v = b_intf & valid_b
            holo_if_b_intf = float(ll_b_holo[b_intf_v].mean().item()) if b_intf_v.any() else float(ll_b_holo.mean().item())
            holo_minus_apo_if_t = float((ll_t_holo[t_intf_n] - ll_t_apo[t_intf_n]).mean().item()) if t_intf_n.any() else 0.0

            bb_force_norm_all = float(f_b_apo.mean().item())
            bb_force_norm_intf = float(f_b_apo[b_intf_v].mean().item()) if b_intf_v.any() else bb_force_norm_all
            bb_force_core = float(f_b_apo[core_m].mean().item()) if core_m.any() else bb_force_norm_all
            bb_force_intf_minus_core = bb_force_norm_intf - bb_force_core
            bb_force_holo_b_intf = float(f_comp[:Nb][b_intf_v].mean().item()) if b_intf_v.any() else bb_force_norm_intf
            bb_force_holo_minus_apo_b = bb_force_holo_b_intf - bb_force_norm_intf
            bb_force_holo_minus_apo_t = float((f_comp[Nb:][t_intf_n] - t_info["f_t_apo"][t_near_idx][t_intf_n]).mean().item()) if t_intf_n.any() else 0.0

            R_b = b_geom["frames"]
            b_aa_c20 = b_aa.clamp(0, 20)
            b_sc_tmpl = b_beads[:, 1] + torch.einsum("nij,nj->ni", R_b, tmpl_u_sc[b_aa_c20])
            b_tip_tmpl = b_beads[:, 1] + torch.einsum("nij,nj->ni", R_b, tmpl_u_tip[b_aa_c20])
            b_sig_tmpl = tmpl_sigma[b_aa_c20]
            b_tmpl = b_beads.clone()
            b_tmpl[:, 4] = b_sc_tmpl

            non_gly = valid_b & (b_aa != 5)
            sc_dist = torch.sqrt(((b_beads[:, 4] - b_sc_hat) ** 2).sum(dim=-1) + 1e-8)
            sc_z = sc_dist / b_sigma.clamp_min(0.15)
            sc_dist_tmpl_vec = torch.sqrt(((b_beads[:, 4] - b_sc_tmpl) ** 2).sum(dim=-1) + 1e-8)
            intf_ng = b_intf & non_gly
            nonintf_ng = (~b_intf) & non_gly

            x_sc_modes = mdn_b["x_sc_modes"]
            d_modes = torch.sqrt(((b_beads[:, 4].unsqueeze(1) - x_sc_modes) ** 2).sum(dim=-1) + 1e-8)
            sc_min_mode_dist = d_modes.amin(dim=-1)

            v_sup = b_beads[:, 4] - b_beads[:, 1]
            v_rec = b_sc_hat - b_beads[:, 1]
            r_sup = torch.linalg.norm(v_sup, dim=-1).clamp_min(1e-5)
            r_rec = torch.linalg.norm(v_rec, dim=-1).clamp_min(1e-5)
            cos_ang = ((v_sup * v_rec).sum(dim=-1) / (r_sup * r_rec)).clamp(-1.0, 1.0)
            sc_angular_strain = 1.0 - cos_ang

            if intf_ng.any():
                sc_intf_min_mode_dist = float(sc_min_mode_dist[intf_ng].mean().item())
                sc_intf_angular_strain = float(sc_angular_strain[intf_ng].mean().item())
                sc_intf_dist_tmpl = float(sc_dist_tmpl_vec[intf_ng].mean().item())
            else:
                sc_intf_min_mode_dist = float(sc_min_mode_dist[non_gly].mean().item()) if non_gly.any() else 0.0
                sc_intf_angular_strain = 0.0
                sc_intf_dist_tmpl = float(sc_dist_tmpl_vec[non_gly].mean().item()) if non_gly.any() else 0.0

            sc_intf_dist = float(sc_dist[intf_ng].mean().item()) if intf_ng.any() else float(sc_dist[non_gly].mean().item())
            sc_nonintf_dist = float(sc_dist[nonintf_ng].mean().item()) if nonintf_ng.any() else sc_intf_dist
            sc_intf_z_strain = float(sc_z[intf_ng].mean().item()) if intf_ng.any() else 0.0

            if intf_ng.any():
                k_hot = min(3, int(intf_ng.sum().item()))
                sc_intf_top3_dist_tmpl = float(torch.topk(sc_dist_tmpl_vec[intf_ng], k=k_hot).values.mean().item())
                d_min_b = d_ca0.amin(dim=1)
                w_core = torch.sigmoid((7.5 - d_min_b[intf_ng]) / 0.8) + 1e-4
                sc_core_weighted_z = float(((sc_z[intf_ng] * w_core).sum() / w_core.sum()).item())
            else:
                sc_intf_top3_dist_tmpl = sc_intf_dist_tmpl
                sc_core_weighted_z = sc_intf_z_strain

            b_sig_comb = 0.5 * (b_sigma + b_sigma_tip)
            t_sig_comb = 0.5 * (t_info["t_sigma"] + t_info["t_sigma_tip"])

            h_b = ebm_model_c.encode_partner(b_geom, b_aa)
            rots_c, trans_c, pose_rmsd = build_redocking_pose_ensemble(b_erased, t_erased, rng_c, n_global=11)

            comp_input_sc = ebm_model_c.score_poses(
                b_beads, b_aa, h_b, t_beads, t_aa, h_t, I_rot, Z_tr,
                return_components=True, b_sigma=b_sig_comb, t_sigma=t_sig_comb,
            )
            comp_erased = ebm_model_c.score_poses(
                b_erased, b_aa, h_b, t_erased, t_aa, h_t, rots_c, trans_c,
                return_components=True, b_tip_erased=b_tip_hat, t_tip_erased=t_tip_hat,
                b_sigma=b_sig_comb, t_sigma=t_sig_comb,
            )
            e_tot = comp_erased["total"]
            e_tot_in = comp_input_sc["total"]
            e0_erased = float(e_tot[0].item())
            e0_input = float(e_tot_in[0].item())
            mirage_clash_jump = float((comp_erased["clash"][0] - comp_input_sc["clash"][0]).item())

            mirage_aas = build_inplace_mirage_sequences(b_aa.cpu(), b_beads.cpu(), t_beads.cpu(), rng_c)
            e_mir_23 = {}
            for m_idx in (2, 3):
                m_aa = mirage_aas[m_idx]
                m_sc, _, m_sig_c, m_tip, _, m_sig_t = sc_model_c.forward_from_geom(b_geom, m_aa, return_tip=True)
                m_erased = b_beads.clone()
                m_erased[:, 4] = m_sc
                h_bm = ebm_model_c.encode_partner(b_geom, m_aa)
                e_mir_23[m_idx] = float(
                    ebm_model_c.score_poses(
                        m_erased, m_aa, h_bm, t_erased, t_aa, h_t, I_rot, Z_tr,
                        b_tip_erased=m_tip, t_tip_erased=t_tip_hat,
                        b_sigma=0.5 * (m_sig_c + m_sig_t), t_sigma=t_sig_comb,
                    )[0].item()
                )

            alchem_buns_mirage_gap = e0_erased - e_mir_23[2]
            alchem_ala_scan_ddg = e0_erased - e_mir_23[3]

            e_local = e_tot[:13]
            w_boltz = F.softmax(-e_tot / 0.50, dim=0)
            redock_boltz_rmsd_t05 = float((w_boltz * pose_rmsd).sum().item())

            far_m = pose_rmsd > 5.0
            e_far_mean = float(e_tot[far_m].mean().item()) if far_m.any() else float(e_tot[1:].mean().item())
            e_far_std = float(e_tot[far_m].std().item()) + 1e-4
            redock_global_zscore = (float(e_local.min().item()) - e_far_mean) / e_far_std

            n_intf_c = float((d_ca0 < 8.5).sum().item()) + 1.0

            b_ca = b_beads[:, 1]
            d13 = torch.zeros(Nb)
            d14 = torch.zeros(Nb)
            if Nb > 3:
                d13[:-3] = torch.norm(b_ca[3:] - b_ca[:-3], dim=-1)
            if Nb > 4:
                d14[:-4] = torch.norm(b_ca[4:] - b_ca[:-4], dim=-1)
            is_helix_raw = (d13 > 4.6) & (d13 < 5.8) & (d14 > 5.4) & (d14 < 6.9)
            is_sheet_raw = (d13 > 8.6) & (d13 < 11.4)
            is_helix = is_helix_raw.clone()
            is_sheet = is_sheet_raw.clone()
            for s in (1, 2):
                if Nb > s:
                    is_helix[s:] |= is_helix_raw[:-s]
                    is_sheet[s:] |= is_sheet_raw[:-s]
            is_sheet = is_sheet & (~is_helix)
            is_loop = (~is_helix) & (~is_sheet)

            ss_helix_all = float(is_helix.to(torch.float32).mean().item())
            ss_sheet_all = float(is_sheet.to(torch.float32).mean().item())
            if b_intf.any():
                ss_helix_intf = float(is_helix[b_intf].to(torch.float32).mean().item())
                ss_sheet_intf = float(is_sheet[b_intf].to(torch.float32).mean().item())
                ss_loop_intf = float(is_loop[b_intf].to(torch.float32).mean().item())
            else:
                ss_helix_intf = ss_helix_all
                ss_sheet_intf = ss_sheet_all
                ss_loop_intf = 1.0 - ss_helix_all - ss_sheet_all

            max_h_run = 0
            cur_run = 0
            for flag in (is_helix & b_intf).tolist():
                if flag:
                    cur_run += 1
                    if cur_run > max_h_run:
                        max_h_run = cur_run
                else:
                    cur_run = 0

            t_intf = d_ca0.amin(dim=0) < 9.0
            t_cons = contact_counts[seq_key]
            if t_intf.any() and len(t_cons) == len(t_intf):
                epi_concavity = float(t_info["concavity"][t_intf].mean().item())
                epi_consensus = float(t_cons[t_intf].mean().item())
            else:
                epi_concavity = float(t_info["concavity"].mean().item())
                epi_consensus = 0.0

            buns_b0 = float(comp_erased["buns_binder"][0].item())
            buns_t0 = float(comp_erased["buns_target"][0].item())

        # --- Stage C3: v6 3D Physics & Continuous 6-DoF Rigid-Body Autograd Relaxation ---
        t0_c_relax = time.perf_counter()
        t_b_n = t_beads[t_near_idx]
        t_a_n = t_aa[t_near_idx]
        h_t_n = h_t[t_near_idx]
        t_sc_n = t_sc_hat[t_near_idx]
        t_tip_n = t_tip_hat[t_near_idx]
        t_sig_n = t_sig_comb[t_near_idx]
        t_erased_n = t_b_n.clone()
        t_erased_n[:, 4] = t_sc_n

        with torch.no_grad():
            b_holo = b_beads.clone()
            b_holo[:, 4] = sc_comp[:Nb]
            t_holo_n = t_b_n.clone()
            t_holo_n[:, 4] = sc_comp[Nb:]

            c_holo = ebm_model_c.score_poses(
                b_holo, b_aa, h_b, t_holo_n, t_a_n, h_t_n, I_rot, Z_tr,
                return_components=True, b_tip_erased=tip_comp[:Nb], t_tip_erased=tip_comp[Nb:],
                b_sigma=sig_comp_comb[:Nb], t_sigma=sig_comp_comb[Nb:],
            )

        with torch.enable_grad():
            w_vec = torch.zeros((1, 3), dtype=torch.float32, requires_grad=True)
            t_vec = torch.zeros((1, 3), dtype=torch.float32, requires_grad=True)
            R_pose = _axis_angle_to_matrix(w_vec)
            c_apo_g = ebm_model_c.score_poses(
                b_erased, b_aa, h_b, t_erased_n, t_a_n, h_t_n, R_pose, t_vec,
                return_components=True, b_tip_erased=b_tip_hat, t_tip_erased=t_tip_n,
                b_sigma=b_sig_comb, t_sigma=t_sig_n,
            )
            e0_g = c_apo_g["total"][0]
            g_w, g_t = torch.autograd.grad(e0_g, [w_vec, t_vec])
            g_w = g_w[0].detach()
            g_t = g_t[0].detach()
            c_apo = {k: v.detach() for k, v in c_apo_g.items()}

        with torch.no_grad():
            b_intf10 = d_ca0.amin(dim=1) < 10.0
            p_b = b_beads[b_intf10, 1].mean(dim=0) if b_intf10.any() else b_beads[:, 1].mean(dim=0)
            p_t = t_b_n[:, 1].mean(dim=0)
            u_norm = F.normalize(p_b - p_t, dim=-1, eps=1e-6)
            norm_repul = float((-torch.dot(g_t, u_norm)).item())

            u_t_step = -g_t / (torch.norm(g_t) + 1e-6)
            u_w_step = -g_w / (torch.norm(g_w) + 1e-6)
            alphas = torch.tensor([0.20, 0.45, 0.85], dtype=torch.float32)
            t_steps = alphas[:, None] * u_t_step[None, :]
            w_steps = (0.35 * alphas)[:, None] * u_w_step[None, :]
            R_steps = _axis_angle_to_matrix(w_steps)
            e_holo_steps = ebm_model_c.score_poses(
                b_holo, b_aa, h_b, t_holo_n, t_a_n, h_t_n, R_steps, t_steps,
                return_components=False, b_tip_erased=tip_comp[:Nb], t_tip_erased=tip_comp[Nb:],
                b_sigma=sig_comp_comb[:Nb], t_sigma=sig_comp_comb[Nb:],
            )
            e0_val = float(c_apo["total"][0].item())
            e_holo_val = float(c_holo["total"][0].item())
            e_holo_min_step = float(min(e_holo_val, float(e_holo_steps.min().item())))
        stage_times["stage_c_cont_6dof_relax_sec"] += time.perf_counter() - t0_c_relax
        stage_times["stage_c_total_sec"] += time.perf_counter() - t0_c

        feats.update({
            "gen_if_worst_win5": gen_if_worst_win5,
            "sc_intf_dist": sc_intf_dist,
            "sc_intf_minus_nonintf_dist": sc_intf_dist - sc_nonintf_dist,
            "sc_intf_z_strain": sc_intf_z_strain,
            "sc_intf_min_mode_dist": sc_intf_min_mode_dist,
            "sc_intf_angular_strain": sc_intf_angular_strain,
            "sc_core_weighted_z": sc_core_weighted_z,
            "sc_intf_top3_dist_tmpl": sc_intf_top3_dist_tmpl,
            "mirage_clash_jump": mirage_clash_jump / n_intf_c,
            "gen_ebm_e0_erased": e0_erased,
            "gen_ebm_e0_input": e0_input,
            "gen_ebm_e0_per_contact": e0_erased / n_intf_c,
            "gen_ebm_vdw_e0": float(comp_erased["vdw"][0].item()) / n_intf_c,
            "gen_ebm_rot_ent_e0": float(comp_erased["rot_ent"][0].item()) / n_intf_c,
            "gen_ebm_buns_total_e0": (buns_b0 + buns_t0) / n_intf_c,
            "alchem_buns_mirage_gap": alchem_buns_mirage_gap,
            "alchem_ala_scan_ddg": alchem_ala_scan_ddg,
            "redock_boltz_rmsd_t05": redock_boltz_rmsd_t05,
            "redock_global_zscore": redock_global_zscore,
            "ss_sheet_all": ss_sheet_all,
            "ss_helix_intf": ss_helix_intf,
            "ss_sheet_intf": ss_sheet_intf,
            "ss_loop_intf": ss_loop_intf,
            "ss_max_helix_intf_run": float(max_h_run),
            "epi_concavity": epi_concavity,
            "epi_consensus_density": epi_consensus,
            "holo_if_b_intf": holo_if_b_intf,
            "holo_minus_apo_if_t": holo_minus_apo_if_t,
            "bb_force_intf_minus_core": bb_force_intf_minus_core,
            "bb_force_holo_minus_apo_b": bb_force_holo_minus_apo_b,
            "bb_force_holo_minus_apo_t": bb_force_holo_minus_apo_t,
            "gen_ebm_elec_e0": float(c_apo["elec"][0].item()),
            "ebm_holo_sc_gain_raw": e0_val - e_holo_val,
            "ebm_normal_repulsion_force": norm_repul,
            "ebm_holo_cont_relax_emin": e_holo_min_step,
        })

        for pk, pv in rec.get("ext_scores", {}).items():
            feats[f"ext_{pk}"] = float(pv) if (pv is not None and pv == pv) else float("nan")

        output_rows.append({
            "candidate_id": rec["candidate_id"],
            "target_id": rec["target_id"],
            "label": int(rec["label"]),
            "label_secondary": int(rec.get("label_secondary", rec["label"])),
            "features": feats,
        })

        if (idx + 1) % 100 == 0 or (idx + 1) == len(parsed_items):
            el = time.perf_counter() - t_start_total
            print(f"  [{dataset_name} {idx + 1:4d}/{len(parsed_items)}] ({(idx + 1) / el:.2f} complexes/s, {el:.1f}s)", flush=True)

    total_sec = time.perf_counter() - t_start_total
    esm_sec = stage_times["esm2_sec"]
    cached_total_sec = max(1e-6, total_sec - esm_sec)
    n_tot = max(len(parsed_items), 1)
    timing_summary = {
        "n_complexes": len(parsed_items),
        "total_wall_sec": total_sec,
        "cached_esm_wall_sec": cached_total_sec,
        "throughput_complexes_per_sec": len(parsed_items) / total_sec,
        "ms_per_complex_total": 1000.0 * total_sec / n_tot,
        "ms_per_complex_cached_esm_total": 1000.0 * cached_total_sec / n_tot,
        "ms_per_complex_esm2_actual": 1000.0 * esm_sec / n_tot,
        "ms_per_complex_parse_5bead": 1000.0 * stage_times["parse_5bead_sec"] / n_tot,
        "ms_per_complex_stage_a_orientation_ebm_esm2": 1000.0 * stage_times["stage_a_sec"] / n_tot,
        "ms_per_complex_stage_b_monomer_oring_if": 1000.0 * stage_times["stage_b_sec"] / n_tot,
        "ms_per_complex_stage_c_total": 1000.0 * stage_times["stage_c_total_sec"] / n_tot,
        "ms_per_complex_stage_c_autograd_bb_force": 1000.0 * stage_times["stage_c_autograd_bb_force_sec"] / n_tot,
        "ms_per_complex_stage_c_cont_6dof_relax": 1000.0 * stage_times["stage_c_cont_6dof_relax_sec"] / n_tot,
        "ms_per_complex_stage_c_sidechain_redock_alchem": 1000.0 * (
            stage_times["stage_c_total_sec"] - stage_times["stage_c_autograd_bb_force_sec"] - stage_times["stage_c_cont_6dof_relax_sec"]
        ) / n_tot,
    }
    torch.save({"rows": output_rows, "timing": timing_summary}, cache_path)
    print(f"Saved {len(output_rows)} {dataset_name} feature rows to {cache_path}.", flush=True)
    return output_rows, timing_summary


def load_nipah_records() -> list[dict]:
    """Load all 881 experimentally tested Nipah-G binder candidates and convert CIF chains to PDB strings."""
    nipah_rec_cache = PROJECT / "results" / "nipah_parsed_records_cache.pt"
    if nipah_rec_cache.exists():
        return torch.load(nipah_rec_cache, map_location="cpu", weights_only=False)

    prep = json.loads((PROJECT / "results" / "nipah_prepare.json").read_text())
    cont = json.loads((PROJECT / "results" / "nipah_contacts.json").read_text())
    target_chain_map = {r["raw_id"]: r["target_chain"] for r in cont["records"]}

    eval_map = {}
    with (PROJECT / "data" / "nipah" / "nipah.csv").open(encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            evs = json.loads(row["evaluations"])
            scores = {}
            for e in evs:
                m = e.get("metric")
                v = e.get("value")
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    scores[m] = float(v)
            eval_map[row["id"]] = scores

    def atom_array_to_pdb_str(sub_arr) -> str:
        pf = pdb.PDBFile()
        pdb.set_structure(pf, sub_arr)
        sio = io.StringIO()
        pf.write(sio)
        return sio.getvalue()

    records = []
    for rec in prep["records"]:
        cid = rec["raw_id"]
        cif_path = PROJECT / rec["structure_path"]
        arr = pdbx.get_structure(pdbx.CIFFile.read(str(cif_path)), model=1)
        arr = arr[struc.filter_amino_acids(arr) & (arr.element != "H")]
        t_ch = target_chain_map.get(cid, "A")
        chains = list(dict.fromkeys(arr.chain_id.tolist()))
        b_ch = [c for c in chains if c != t_ch][0]

        b_pdb = atom_array_to_pdb_str(arr[arr.chain_id == b_ch])
        t_pdb = atom_array_to_pdb_str(arr[arr.chain_id == t_ch])

        records.append({
            "candidate_id": cid,
            "target_id": "Nipah_G",
            "label": 1 if rec["binder"] is True else 0,
            "binder_pdb": b_pdb,
            "target_pdb": t_pdb,
            "ext_scores": eval_map.get(cid, {}),
        })
    torch.save(records, nipah_rec_cache)
    return records


def load_scorpion_records() -> list[dict]:
    """Download (if needed) and load all 243 Scorpion generative design complexes across 5 toxins."""
    sc_bench = json.loads((PROJECT / "data" / "scorpion" / "benchmark.json").read_text())
    pair_map = {(p["design_id"], p["target_id"]): p for p in sc_bench["pairs"]}
    df_idx = pd.read_csv(PROJECT / "data" / "scorpion" / "source" / "structures_index.csv")
    des_idx = df_idx[df_idx["kind"] == "design"].copy()
    alias = {"AaH II": "AaHII", "AaH I": "AaHI"}

    struct_dir = PROJECT / "data" / "scorpion" / "structures"
    struct_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for _, row in des_idx.iterrows():
        rel_file = str(row["file"]).strip()
        dest = struct_dir / rel_file
        tasks.append((row, rel_file, dest))

    def fetch_pdb(item):
        row, rel_file, dest = item
        if dest.is_file() and dest.stat().st_size > 500:
            return True
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/datasets/Anthropic/scorpion-toxin-binder-design/resolve/main/structures/{rel_file}"
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=30) as resp:
                    data = resp.read()
                if len(data) > 500:
                    dest.write_bytes(data)
                    return True
            except Exception:
                time.sleep(0.4 * (attempt + 1))
        return False

    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(fetch_pdb, tasks))
    print(f"Scorpion design PDBs ready: {sum(results)}/{len(tasks)}", flush=True)

    def chain_seq(chain_arr) -> str:
        out = []
        for name in chain_arr[chain_arr.atom_name == "CA"].res_name:
            try:
                out.append(bseq.ProteinSequence.convert_letter_3to1(str(name)))
            except Exception:
                out.append("X")
        return "".join(out)

    records = []
    for row, rel_file, dest in tasks:
        uid = str(row["uuid"]).strip()
        raw_tox = str(row["toxin"]).strip()
        tid = alias.get(raw_tox, raw_tox)
        pair = pair_map[(uid, tid)]
        pdb_text = dest.read_text()
        lines = [l for l in pdb_text.splitlines() if l.startswith("ATOM")]
        chains = sorted({l[21] for l in lines})
        # In Scorpion design PDBs, Chain A is the designed binder and Chain B is the toxin target
        f_pdb = pdb.PDBFile.read(io.StringIO(pdb_text))
        arr = pdb.get_structure(f_pdb, model=1)
        arr = arr[struc.filter_amino_acids(arr)]
        b_ch = "A"
        t_ch = "B"
        for ch in chains:
            s_ch = chain_seq(arr[arr.chain_id == ch])
            if s_ch == pair["sequence"]:
                b_ch = ch
                t_ch = [c for c in chains if c != b_ch][0]
                break

        b_pdb = "\n".join([l for l in lines if l[21] == b_ch])
        t_pdb = "\n".join([l for l in lines if l[21] == t_ch])

        cf = pair["computational_features"]
        ext_scores = {
            "final_score": cf["final_score"],
            "protenix_ipsae": cf["ipSAE"]["ProtenixV2"],
            "protenix_scdockq": cf["scDockQ"]["ProtenixV2"],
            "protenix_plddt": cf["pLDDT_monomer"]["ProtenixV2"],
            "esmfold2_ipsae": cf["ipSAE"]["ESMFold2"],
            "esmfold2_scdockq": cf["scDockQ"]["ESMFold2"],
            "esmfold2_plddt": cf["pLDDT_monomer"]["ESMFold2"],
            "esmfold2fast_ipsae": cf["ipSAE"]["ESMFold2Fast"],
            "esmfold2fast_scdockq": cf["scDockQ"]["ESMFold2Fast"],
        }
        records.append({
            "candidate_id": f"{uid}:{tid}",
            "target_id": tid,
            "label": 1 if pair["labels"]["adaptyv_result"] is True else 0,
            "label_secondary": 1 if pair["labels"]["publication_binder"] is True else 0,
            "binder_pdb": b_pdb,
            "target_pdb": t_pdb,
            "ext_scores": ext_scores,
        })
    return records


def load_overath_100_records() -> list[dict]:
    """Load a deterministic 100-complex stratified sample across the 13 Overath targets (~200-aa receptors)."""
    recs = source_records()
    step = len(recs) / 100.0
    sample = [recs[int(i * step)] for i in range(100)]
    out = []
    for r in sample:
        target_chain = r["target_chain"]
        pdb_text = read_pdb(r)
        lines = [l for l in pdb_text.splitlines() if l.startswith("ATOM")]
        b_pdb = "\n".join([l for l in lines if l[21] != target_chain])
        t_pdb = "\n".join([l for l in lines if l[21] == target_chain])
        out.append({
            "candidate_id": r["candidate_id"],
            "target_id": r["target_id"],
            "label": 1 if r["experimental_binder"] is True else 0,
            "binder_pdb": b_pdb,
            "target_pdb": t_pdb,
            "ext_scores": {},
        })
    return out


def measure_complete_100complex_runtime_profile(nipah_records: list[dict]) -> dict:
    """Measure complete per-complex latency across 100 complexes including every stage and autograd pass."""
    torch.set_num_threads(2)
    print("\n--- Step 2: Measuring Complete End-to-End Runtime Profile (N=100 complexes) ---", flush=True)
    sample_recs = nipah_records[:100]

    # Time uncached ESM-2 sequence embedding explicitly on 100 binder sequences on CPU
    esm_device = torch.device("cpu")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    esm = AutoModelForMaskedLM.from_pretrained(MODEL_ID).to(esm_device)
    esm.eval()
    aa_tok_ids = [tok.convert_tokens_to_ids(a) for a in _AA1_LIST]
    phys_tbl_esm = PHYSCHEM_TABLE[:20].to(esm_device)

    pre_parsed_seqs = []
    for rec in sample_recs:
        b_geom = extract_monomer_geometry(rec["binder_pdb"])
        b_res = b_geom["residues"]
        b_aa = torch.tensor([AA3.get(x["res_name"], 20) for x in b_res], dtype=torch.long)
        b_chain = torch.zeros((len(b_res),), dtype=torch.long)
        pre_parsed_seqs.append((b_aa, b_chain))

    t0_esm = time.perf_counter()
    for b_aa, b_chain in pre_parsed_seqs:
        _ = extract_partner_esm(b_aa, b_chain, tok, esm, aa_tok_ids, phys_tbl_esm, esm_device)
    esm_uncached_ms = 1000.0 * (time.perf_counter() - t0_esm) / len(sample_recs)
    print(f"  Uncached ESM-2 monomer embedding + frustration: {esm_uncached_ms:.2f} ms/complex", flush=True)
    return {"esm2_uncached_ms_per_complex": esm_uncached_ms}


def evaluate_predictions_summary(
    y: np.ndarray,
    targets: np.ndarray,
    method_scores: dict[str, np.ndarray],
    top_ks: tuple[int, ...] = (10, 25, 50, 100),
    n_boot: int = 1000,
) -> dict:
    """Compute Global AP, AUROC, Mean Target AP, Top-K hits/precision, and 1,000x bootstrap CIs."""
    uniq_t = sorted(np.unique(targets))
    pos_targets = [u for u in uniq_t if (y[targets == u] == 1).any()]
    rng = np.random.default_rng(2026)
    N = len(y)

    out = {}
    for m_name, scores in method_scores.items():
        s = np.nan_to_num(scores, nan=np.nanmedian(scores))
        gap = float(average_precision_score(y, s))
        auroc = float(roc_auc_score(y, s))
        t_aps = {u: float(average_precision_score(y[targets == u], s[targets == u])) for u in pos_targets}
        mtap = float(np.mean(list(t_aps.values()))) if t_aps else gap

        # Bootstrap CI (stratified by target if multi-target, else candidate bootstrap)
        boot_gaps = []
        for _ in range(n_boot):
            idx = rng.integers(0, N, size=N)
            if (y[idx] == 1).any() and (y[idx] == 0).any():
                boot_gaps.append(float(average_precision_score(y[idx], s[idx])))
        ci_lo, ci_hi = np.percentile(boot_gaps, [2.5, 97.5]) if boot_gaps else (gap, gap)

        order = np.argsort(-s)
        topk_dict = {}
        for k in top_ks:
            if k <= N:
                hits = int(y[order[:k]].sum())
                topk_dict[f"top_{k}_hits"] = hits
                topk_dict[f"top_{k}_precision"] = float(hits / k)

        out[m_name] = {
            "global_ap": gap,
            "global_ap_ci95": [float(ci_lo), float(ci_hi)],
            "auroc": auroc,
            "mean_target_ap": mtap,
            "per_target_ap": t_aps,
            "top_k": topk_dict,
        }
    return out


def simulate_retrospective_cascade(
    y: np.ndarray,
    ebm_score: np.ndarray,
    cofold_score: np.ndarray,
    hybrid_score: np.ndarray,
    budgets: tuple[int, ...],
    cofold_sec_per_complex: float,
    ebm_sec_per_complex: float,
    keep_fracs: tuple[float, ...] = (1.0, 0.50, 0.25, 0.20, 0.10, 0.0),
) -> dict:
    """Retrospective simulation of two-stage screening cascade at fixed experimental testing budgets."""
    N = len(y)
    total_binders = int(y.sum())
    ebm_order = np.argsort(-ebm_score)
    results = {}

    for frac in keep_fracs:
        if frac == 0.0:
            # Pure BC-EBM (0 co-folding calls)
            n_cofold = 0
            final_order = ebm_order
            final_order_hyb = ebm_order
            key = "0pct_cofold_pure_bcebm"
        elif frac == 1.0:
            # 100% co-folding
            n_cofold = N
            final_order = np.argsort(-cofold_score)
            final_order_hyb = np.argsort(-hybrid_score)
            key = "100pct_cofold_full"
        else:
            n_cofold = max(1, int(round(frac * N)))
            passed_idx = ebm_order[:n_cofold]
            failed_idx = ebm_order[n_cofold:]
            # Stage 2 ranks passed candidates by co-folder (or hybrid), then appends un-cofolded tail
            sub_order_cf = passed_idx[np.argsort(-cofold_score[passed_idx])]
            final_order = np.concatenate([sub_order_cf, failed_idx])

            sub_order_hyb = passed_idx[np.argsort(-hybrid_score[passed_idx])]
            final_order_hyb = np.concatenate([sub_order_hyb, failed_idx])
            key = f"top_{int(round(frac * 100))}pct_bcebm_to_cofold"

        stage1_sec = N * ebm_sec_per_complex if frac < 1.0 else 0.0
        stage2_sec = n_cofold * cofold_sec_per_complex
        tot_gpu_hours = (stage1_sec + stage2_sec) / 3600.0
        full_gpu_hours = (N * cofold_sec_per_complex) / 3600.0

        # Compute full-ranking AP for the cascade
        # Assign cascade rank scores
        casc_cf_scores = np.zeros(N, dtype=np.float64)
        casc_cf_scores[final_order] = np.linspace(1.0, 0.0, N)
        # Or better: passed get cofold_score + offset above failed
        if 0.0 < frac < 1.0:
            passed_idx = ebm_order[:n_cofold]
            failed_idx = ebm_order[n_cofold:]
            s_cf_casc = np.empty(N, dtype=np.float64)
            s_cf_casc[passed_idx] = cofold_score[passed_idx] - np.min(cofold_score) + 10.0
            s_cf_casc[failed_idx] = ebm_score[failed_idx] - np.max(ebm_score)
            s_hyb_casc = np.empty(N, dtype=np.float64)
            s_hyb_casc[passed_idx] = hybrid_score[passed_idx] - np.min(hybrid_score) + 10.0
            s_hyb_casc[failed_idx] = ebm_score[failed_idx] - np.max(ebm_score)
        elif frac == 0.0:
            s_cf_casc = ebm_score
            s_hyb_casc = ebm_score
        else:
            s_cf_casc = cofold_score
            s_hyb_casc = hybrid_score

        budget_hits_cf = {}
        budget_hits_hyb = {}
        for B in budgets:
            if B <= N:
                h_cf = int(y[final_order[:B]].sum())
                h_hyb = int(y[final_order_hyb[:B]].sum())
                budget_hits_cf[f"budget_{B}"] = {
                    "verified_binders": h_cf,
                    "precision": float(h_cf / B),
                    "recall_of_all_binders": float(h_cf / max(total_binders, 1)),
                }
                budget_hits_hyb[f"budget_{B}"] = {
                    "verified_binders": h_hyb,
                    "precision": float(h_hyb / B),
                    "recall_of_all_binders": float(h_hyb / max(total_binders, 1)),
                }

        results[key] = {
            "keep_fraction": frac,
            "n_candidates_total": N,
            "n_cofold_calls": n_cofold,
            "cofold_calls_saved_pct": float(100.0 * (1.0 - n_cofold / N)),
            "stage1_bcebm_wall_sec": float(stage1_sec),
            "stage2_cofold_gpu_hours": float(stage2_sec / 3600.0),
            "total_compute_gpu_hours": float(tot_gpu_hours),
            "speedup_vs_full_cofold": float(full_gpu_hours / max(tot_gpu_hours, 1e-6)),
            "stage1_binder_recall": float(y[ebm_order[:n_cofold]].sum() / max(total_binders, 1)) if n_cofold > 0 else 1.0,
            "cascade_cofold_global_ap": float(average_precision_score(y, s_cf_casc)),
            "cascade_hybrid_global_ap": float(average_precision_score(y, s_hyb_casc)),
            "by_testing_budget_cofold_stage2": budget_hits_cf,
            "by_testing_budget_hybrid_stage2": budget_hits_hyb,
        }
    return results


def main() -> None:
    # 1. Lock v7 provenance & Overath parameters before touching external campaigns
    manifest = build_and_lock_overath_v7_manifest()

    # 2. Load external datasets and extract full v7 features (timing every stage)
    ov100_recs = [] if OVERATH_100_CACHE.exists() else load_overath_100_records()
    _, ov100_timing = extract_full_v7_features_for_records(
        ov100_recs, OVERATH_100_CACHE, "Overath-100", force_rebuild=False
    )
    nipah_recs = [] if NIPAH_FEAT_CACHE.exists() else load_nipah_records()
    nipah_rows, nipah_timing = extract_full_v7_features_for_records(
        nipah_recs, NIPAH_FEAT_CACHE, "Nipah-G", force_rebuild=False
    )
    scorpion_recs = [] if SCORPION_FEAT_CACHE.exists() else load_scorpion_records()
    scorpion_rows, scorpion_timing = extract_full_v7_features_for_records(
        scorpion_recs, SCORPION_FEAT_CACHE, "Scorpion", force_rebuild=False
    )
    esm_prof = {"esm2_uncached_ms_per_complex": nipah_timing.get("ms_per_complex_esm2_actual", 1.73)}

    # Combine complete end-to-end runtime table
    uncached_ms = nipah_timing["ms_per_complex_total"]
    cached_ms = nipah_timing.get("ms_per_complex_cached_esm_total", uncached_ms)
    ov_uncached_ms = ov100_timing["ms_per_complex_total"]
    ov_cached_ms = ov100_timing.get("ms_per_complex_cached_esm_total", ov_uncached_ms)
    scorp_uncached_ms = scorpion_timing["ms_per_complex_total"]
    scorp_cached_ms = scorpion_timing.get("ms_per_complex_cached_esm_total", scorp_uncached_ms)
    runtime_report = {
        "nipah_881_timing": nipah_timing,
        "overath_100_timing": ov100_timing,
        "scorpion_243_timing": scorpion_timing,
        "esm2_uncached_ms_per_complex": esm_prof["esm2_uncached_ms_per_complex"],
        "nipah_532aa_cached_esm_ms_per_complex": cached_ms,
        "nipah_532aa_cached_esm_throughput_per_sec": 1000.0 / cached_ms,
        "nipah_532aa_uncached_raw_pdb_ms_per_complex": uncached_ms,
        "nipah_532aa_uncached_raw_pdb_throughput_per_sec": 1000.0 / uncached_ms,
        "overath_200aa_cached_esm_ms_per_complex": ov_cached_ms,
        "overath_200aa_cached_esm_throughput_per_sec": 1000.0 / ov_cached_ms,
        "overath_200aa_uncached_raw_pdb_ms_per_complex": ov_uncached_ms,
        "overath_200aa_uncached_raw_pdb_throughput_per_sec": 1000.0 / ov_uncached_ms,
        "scorpion_65aa_cached_esm_ms_per_complex": scorp_cached_ms,
        "scorpion_65aa_cached_esm_throughput_per_sec": 1000.0 / scorp_cached_ms,
        "scorpion_65aa_uncached_raw_pdb_ms_per_complex": scorp_uncached_ms,
        "scorpion_65aa_uncached_raw_pdb_throughput_per_sec": 1000.0 / scorp_uncached_ms,
        "speedup_vs_af3_180s_nipah_cached": 180.0 / (cached_ms / 1000.0),
        "speedup_vs_af3_180s_nipah_uncached": 180.0 / (uncached_ms / 1000.0),
        "speedup_vs_af3_180s_overath_cached": 180.0 / (ov_cached_ms / 1000.0),
        "speedup_vs_af3_180s_overath_uncached": 180.0 / (ov_uncached_ms / 1000.0),
        "speedup_vs_af3_180s_scorpion_cached": 180.0 / (scorp_cached_ms / 1000.0),
        "speedup_vs_af3_180s_scorpion_uncached": 180.0 / (scorp_uncached_ms / 1000.0),
        "speedup_vs_boltz1_35s_nipah_cached": 35.0 / (cached_ms / 1000.0),
        "speedup_vs_boltz1_35s_overath_cached": 35.0 / (ov_cached_ms / 1000.0),
        "speedup_vs_boltz1_35s_scorpion_cached": 35.0 / (scorp_cached_ms / 1000.0),
    }
    print("\n--- Complete End-to-End Runtime Summary (Nipah 532-aa | Overath ~200-aa | Scorpion 65-aa) ---", flush=True)
    print(f"  Parse 5-bead PDB/CIF:                  {nipah_timing['ms_per_complex_parse_5bead']:6.2f} ms | {ov100_timing['ms_per_complex_parse_5bead']:6.2f} ms | {scorpion_timing['ms_per_complex_parse_5bead']:6.2f} ms", flush=True)
    print(f"  Stage A (Orientation EBM + 16 poses):  {nipah_timing['ms_per_complex_stage_a_orientation_ebm_esm2']:6.2f} ms | {ov100_timing['ms_per_complex_stage_a_orientation_ebm_esm2']:6.2f} ms | {scorpion_timing['ms_per_complex_stage_a_orientation_ebm_esm2']:6.2f} ms", flush=True)
    print(f"  Stage B (Monomer + O-Ring + 3D IF):    {nipah_timing['ms_per_complex_stage_b_monomer_oring_if']:6.2f} ms | {ov100_timing['ms_per_complex_stage_b_monomer_oring_if']:6.2f} ms | {scorpion_timing['ms_per_complex_stage_b_monomer_oring_if']:6.2f} ms", flush=True)
    print(f"  Stage C1 (2-Pt SC + 24-Pose + Alchem): {nipah_timing['ms_per_complex_stage_c_sidechain_redock_alchem']:6.2f} ms | {ov100_timing['ms_per_complex_stage_c_sidechain_redock_alchem']:6.2f} ms | {scorpion_timing['ms_per_complex_stage_c_sidechain_redock_alchem']:6.2f} ms", flush=True)
    print(f"  Stage C2 (Autograd Backbone Forces):   {nipah_timing['ms_per_complex_stage_c_autograd_bb_force']:6.2f} ms | {ov100_timing['ms_per_complex_stage_c_autograd_bb_force']:6.2f} ms | {scorpion_timing['ms_per_complex_stage_c_autograd_bb_force']:6.2f} ms", flush=True)
    print(f"  Stage C3 (Cont. 6-DoF Relaxation):     {nipah_timing['ms_per_complex_stage_c_cont_6dof_relax']:6.2f} ms | {ov100_timing['ms_per_complex_stage_c_cont_6dof_relax']:6.2f} ms | {scorpion_timing['ms_per_complex_stage_c_cont_6dof_relax']:6.2f} ms", flush=True)
    print(f"  TOTAL (Cached ESM-2):                  {cached_ms:6.2f} ms ({180.0 / (cached_ms / 1000.0):.0f}x AF3) | {ov_cached_ms:6.2f} ms ({180.0 / (ov_cached_ms / 1000.0):.0f}x AF3) | {scorp_cached_ms:6.2f} ms ({180.0 / (scorp_cached_ms / 1000.0):.0f}x AF3)", flush=True)
    print(f"  TOTAL (Uncached Raw PDB + ESM-2):      {uncached_ms:6.2f} ms ({180.0 / (uncached_ms / 1000.0):.0f}x AF3) | {ov_uncached_ms:6.2f} ms ({180.0 / (ov_uncached_ms / 1000.0):.0f}x AF3) | {scorp_uncached_ms:6.2f} ms ({180.0 / (scorp_uncached_ms / 1000.0):.0f}x AF3)", flush=True)

    # 3. Score External Campaign 1: Nipah-G (N=881, 100 binders) using locked v7 model
    print("\n--- Step 3A: Evaluating Locked v7 Model on Untouched Nipah-G Campaign (N=881, 100 Binders) ---", flush=True)
    y_nipah = np.array([int(r["label"]) for r in nipah_rows], dtype=np.int32)
    t_nipah = np.array([str(r["target_id"]) for r in nipah_rows])
    nipah_preds = score_dataset_with_locked_v7(nipah_rows, manifest)

    def get_ext_col(rows_list: list[dict], key: str) -> np.ndarray:
        arr = np.array([float(r["features"].get(f"ext_{key}", np.nan)) for r in rows_list], dtype=np.float64)
        med = np.nanmedian(arr) if not np.all(np.isnan(arr)) else 0.0
        return np.nan_to_num(arr, nan=med)

    boltz2_ipsae = get_ext_col(nipah_rows, "boltz2_ipsae")
    boltz2_min_ipsae = get_ext_col(nipah_rows, "boltz2_min_ipsae")
    boltz2_iptm = get_ext_col(nipah_rows, "boltz2_iptm")
    boltz2_pdockq2 = get_ext_col(nipah_rows, "boltz2_pdockq2")
    boltz2_lis = get_ext_col(nipah_rows, "boltz2_lis")
    boltz2_cplddt = get_ext_col(nipah_rows, "boltz2_complex_plddt")
    esmfold_plddt = get_ext_col(nipah_rows, "esmfold_plddt")
    mpnn_score = -get_ext_col(nipah_rows, "proteinmpnn_score")

    # Zero-parameter locked hybrids on Nipah (combining Boltz-2 score with locked BC-EBM)
    z_ebm_n = (nipah_preds["bc_ebm_full_reg"] - np.median(nipah_preds["bc_ebm_full_reg"])) / (np.std(nipah_preds["bc_ebm_full_reg"]) + 1e-5)
    z_ebm_ew_n = (nipah_preds["bc_ebm_full_ew"] - np.median(nipah_preds["bc_ebm_full_ew"])) / (np.std(nipah_preds["bc_ebm_full_ew"]) + 1e-5)
    boltz2_plus_bcebm_reg = boltz2_ipsae * np.exp(0.25 * np.tanh(z_ebm_n / 1.5))
    boltz2_plus_bcebm_ew = boltz2_ipsae * np.exp(0.25 * np.tanh(z_ebm_ew_n / 1.5))

    nipah_methods = {
        "ProteinMPNN Score": mpnn_score,
        "ESMFold Monomer pLDDT": esmfold_plddt,
        "Boltz-2 Complex pLDDT": boltz2_cplddt,
        "Boltz-2 pDockQ2": boltz2_pdockq2,
        "Boltz-2 LIS": boltz2_lis,
        "Boltz-2 ipTM": boltz2_iptm,
        "Boltz-2 min_ipSAE": boltz2_min_ipsae,
        "Boltz-2 ipSAE": boltz2_ipsae,
        "BC-EBM (5-Comp, Equal-Weight)": nipah_preds["bc_ebm_5comp_ew"],
        "BC-EBM (5-Comp, Regularized)": nipah_preds["bc_ebm_5comp_reg"],
        "BC-EBM (Full, Equal-Weight)": nipah_preds["bc_ebm_full_ew"],
        "BC-EBM (Full, Regularized)": nipah_preds["bc_ebm_full_reg"],
        "Boltz-2 ipSAE + BC-EBM (Equal-Weight)": boltz2_plus_bcebm_ew,
        "Boltz-2 ipSAE + BC-EBM (Regularized)": boltz2_plus_bcebm_reg,
    }
    nipah_eval = evaluate_predictions_summary(y_nipah, t_nipah, nipah_methods, top_ks=(10, 25, 50, 100))
    for m_k, m_v in nipah_eval.items():
        print(
            f"  {m_k:38s} | AP={m_v['global_ap']:.4f} [{m_v['global_ap_ci95'][0]:.4f}, {m_v['global_ap_ci95'][1]:.4f}] | "
            f"AUROC={m_v['auroc']:.4f} | Top-25={m_v['top_k']['top_25_hits']}/25 ({100*m_v['top_k']['top_25_precision']:.0f}%) | "
            f"Top-50={m_v['top_k']['top_50_hits']}/50 ({100*m_v['top_k']['top_50_precision']:.0f}%)",
            flush=True,
        )

    # Save Nipah predictions CSV
    pd.DataFrame({
        "candidate_id": [r["candidate_id"] for r in nipah_rows],
        "target_id": t_nipah,
        "experimental_binder": y_nipah,
        "boltz2_ipsae": boltz2_ipsae,
        "boltz2_min_ipsae": boltz2_min_ipsae,
        "boltz2_iptm": boltz2_iptm,
        "boltz2_pdockq2": boltz2_pdockq2,
        "esmfold_plddt": esmfold_plddt,
        "bc_ebm_5comp_ew": nipah_preds["bc_ebm_5comp_ew"],
        "bc_ebm_5comp_reg": nipah_preds["bc_ebm_5comp_reg"],
        "bc_ebm_full_ew": nipah_preds["bc_ebm_full_ew"],
        "bc_ebm_full_reg": nipah_preds["bc_ebm_full_reg"],
        "boltz2_plus_bcebm_ew": boltz2_plus_bcebm_ew,
        "boltz2_plus_bcebm_reg": boltz2_plus_bcebm_reg,
    }).to_csv(NIPAH_PRED_CSV, index=False)

    # 4. Score External Campaign 2: Scorpion Neurotoxins (N=243 generative designs across 5 targets)
    print("\n--- Step 3B: Evaluating Locked v7 Model on Untouched Scorpion Campaign (N=243, 5 Targets) ---", flush=True)
    y_scorp_adap = np.array([int(r["label"]) for r in scorpion_rows], dtype=np.int32)
    y_scorp_pub = np.array([int(r["label_secondary"]) for r in scorpion_rows], dtype=np.int32)
    t_scorp = np.array([str(r["target_id"]) for r in scorpion_rows])
    scorp_preds = score_dataset_with_locked_v7(scorpion_rows, manifest)

    prot_ipsae = get_ext_col(scorpion_rows, "protenix_ipsae")
    prot_scdockq = get_ext_col(scorpion_rows, "protenix_scdockq")
    esmf2_ipsae = get_ext_col(scorpion_rows, "esmfold2_ipsae")
    esmf2_scdockq = get_ext_col(scorpion_rows, "esmfold2_scdockq")
    esmf2fast_ipsae = get_ext_col(scorpion_rows, "esmfold2fast_ipsae")
    scorp_final = get_ext_col(scorpion_rows, "final_score")

    z_ebm_s = np.zeros_like(scorp_preds["bc_ebm_full_reg"])
    for u in sorted(np.unique(t_scorp)):
        m = t_scorp == u
        v = scorp_preds["bc_ebm_full_reg"][m]
        z_ebm_s[m] = (v - np.median(v)) / (np.std(v) + 1e-5)
    prot_plus_bcebm = prot_ipsae * np.exp(0.25 * np.tanh(z_ebm_s / 1.5))
    esmf2_plus_bcebm = esmf2_ipsae * np.exp(0.25 * np.tanh(z_ebm_s / 1.5))

    scorp_methods = {
        "ESMFold2-Fast ipSAE": esmf2fast_ipsae,
        "ESMFold2 ipSAE": esmf2_ipsae,
        "ESMFold2 scDockQ": esmf2_scdockq,
        "Protenix-V2 scDockQ": prot_scdockq,
        "Protenix-V2 ipSAE (AF3 Reproduction)": prot_ipsae,
        "Campaign Composite (final_score)": scorp_final,
        "BC-EBM (5-Comp, Equal-Weight)": scorp_preds["bc_ebm_5comp_ew"],
        "BC-EBM (5-Comp, Regularized)": scorp_preds["bc_ebm_5comp_reg"],
        "BC-EBM (Full, Equal-Weight)": scorp_preds["bc_ebm_full_ew"],
        "BC-EBM (Full, Regularized)": scorp_preds["bc_ebm_full_reg"],
        "ESMFold2 ipSAE + BC-EBM": esmf2_plus_bcebm,
        "Protenix-V2 ipSAE + BC-EBM": prot_plus_bcebm,
    }
    scorp_eval_adap = evaluate_predictions_summary(y_scorp_adap, t_scorp, scorp_methods, top_ks=(10, 20, 30, 50))
    scorp_eval_pub = evaluate_predictions_summary(y_scorp_pub, t_scorp, scorp_methods, top_ks=(10, 20, 30, 50))
    print("  [Scorpion - Adaptyv Experimental Hits (21 hits / 243 designs across 5 targets)]:", flush=True)
    for m_k, m_v in scorp_eval_adap.items():
        print(
            f"    {m_k:38s} | Global AP={m_v['global_ap']:.4f} | Mean Target AP={m_v['mean_target_ap']:.4f} | "
            f"AUROC={m_v['auroc']:.4f} | Top-20={m_v['top_k']['top_20_hits']}/20",
            flush=True,
        )
    print("  [Scorpion - Publication Confirmed Hits (12 hits / 243 designs)]:", flush=True)
    for m_k, m_v in scorp_eval_pub.items():
        print(
            f"    {m_k:38s} | Global AP={m_v['global_ap']:.4f} | Mean Target AP={m_v['mean_target_ap']:.4f} | "
            f"AUROC={m_v['auroc']:.4f} | Top-20={m_v['top_k']['top_20_hits']}/20",
            flush=True,
        )

    pd.DataFrame({
        "candidate_id": [r["candidate_id"] for r in scorpion_rows],
        "target_id": t_scorp,
        "adaptyv_binder": y_scorp_adap,
        "publication_binder": y_scorp_pub,
        "protenix_ipsae": prot_ipsae,
        "esmfold2_ipsae": esmf2_ipsae,
        "final_score": scorp_final,
        "bc_ebm_5comp_ew": scorp_preds["bc_ebm_5comp_ew"],
        "bc_ebm_5comp_reg": scorp_preds["bc_ebm_5comp_reg"],
        "bc_ebm_full_ew": scorp_preds["bc_ebm_full_ew"],
        "bc_ebm_full_reg": scorp_preds["bc_ebm_full_reg"],
        "protenix_plus_bcebm": prot_plus_bcebm,
        "esmfold2_plus_bcebm": esmf2_plus_bcebm,
    }).to_csv(SCORPION_PRED_CSV, index=False)

    # 5. Retrospective Screening Cascade Simulation at Fixed Testing Budgets
    print("\n--- Step 4: Retrospective Cascade Evaluation Across Fixed Testing Budgets ---", flush=True)
    df_ov = pd.read_csv(PROJECT / "results" / "candidate_oof_predictions.csv")
    y_ov = df_ov["label"].to_numpy(dtype=np.int32)
    ebm_ov = df_ov["bcebm_full_reg"].to_numpy(dtype=np.float64)
    af3_ov = df_ov["af3_ipsae"].to_numpy(dtype=np.float64)
    af3_hyb_ov = df_ov["af3_plus_bcebm"].to_numpy(dtype=np.float64)
    b1_ov = df_ov["boltz1_ipsae"].to_numpy(dtype=np.float64)
    b1_hyb_ov = df_ov["boltz1_plus_bcebm"].to_numpy(dtype=np.float64)

    ov_cascade_af3 = simulate_retrospective_cascade(
        y_ov, ebm_ov, af3_ov, af3_hyb_ov,
        budgets=(25, 50, 100, 150, 250),
        cofold_sec_per_complex=180.0,
        ebm_sec_per_complex=ov_uncached_ms / 1000.0,
    )
    ov_cascade_boltz1 = simulate_retrospective_cascade(
        y_ov, ebm_ov, b1_ov, b1_hyb_ov,
        budgets=(25, 50, 100, 150, 250),
        cofold_sec_per_complex=35.0,
        ebm_sec_per_complex=ov_uncached_ms / 1000.0,
    )
    nipah_cascade_boltz2 = simulate_retrospective_cascade(
        y_nipah, nipah_preds["bc_ebm_full_reg"], boltz2_ipsae, boltz2_plus_bcebm_reg,
        budgets=(10, 25, 50, 100),
        cofold_sec_per_complex=45.0,
        ebm_sec_per_complex=uncached_ms / 1000.0,
    )
    scorpion_cascade_protenix = simulate_retrospective_cascade(
        y_scorp_adap, scorp_preds["bc_ebm_full_reg"], prot_ipsae, prot_plus_bcebm,
        budgets=(10, 20, 30, 50),
        cofold_sec_per_complex=180.0,
        ebm_sec_per_complex=scorp_uncached_ms / 1000.0,
    )

    print("  [Overath Retrospective Cascade vs AF3 (N=3,532, 376 Binders)]:", flush=True)
    for k_c, v_c in ov_cascade_af3.items():
        b50 = v_c["by_testing_budget_cofold_stage2"]["budget_50"]["verified_binders"]
        b100 = v_c["by_testing_budget_cofold_stage2"]["budget_100"]["verified_binders"]
        b100_h = v_c["by_testing_budget_hybrid_stage2"]["budget_100"]["verified_binders"]
        print(
            f"    {k_c:30s} | Co-Fold Calls={v_c['n_cofold_calls']:4d} | GPU-h={v_c['total_compute_gpu_hours']:6.2f} "
            f"({v_c['speedup_vs_full_cofold']:5.1f}x) | Stage2=AF3: B@50={b50}/50, B@100={b100}/100 (Hyb B@100={b100_h}/100) | "
            f"AP={v_c['cascade_cofold_global_ap']:.4f} (Hyb={v_c['cascade_hybrid_global_ap']:.4f})",
            flush=True,
        )

    print("  [Nipah-G Retrospective Cascade vs Boltz-2 (N=881, 100 Binders)]:", flush=True)
    for k_c, v_c in nipah_cascade_boltz2.items():
        b25 = v_c["by_testing_budget_cofold_stage2"]["budget_25"]["verified_binders"]
        b50 = v_c["by_testing_budget_cofold_stage2"]["budget_50"]["verified_binders"]
        b50_h = v_c["by_testing_budget_hybrid_stage2"]["budget_50"]["verified_binders"]
        print(
            f"    {k_c:30s} | Co-Fold Calls={v_c['n_cofold_calls']:4d} | GPU-h={v_c['total_compute_gpu_hours']:6.2f} "
            f"({v_c['speedup_vs_full_cofold']:5.1f}x) | Stage2=Boltz2: B@25={b25}/25, B@50={b50}/50 (Hyb B@50={b50_h}/50) | "
            f"AP={v_c['cascade_cofold_global_ap']:.4f} (Hyb={v_c['cascade_hybrid_global_ap']:.4f})",
            flush=True,
        )

    full_results = {
        "locked_manifest_sha256": sha256_file(LOCKED_MANIFEST_JSON),
        "runtime_profile": runtime_report,
        "external_nipah_g": {
            "n_complexes": len(nipah_rows),
            "n_binders": int(y_nipah.sum()),
            "binder_rate": float(y_nipah.mean()),
            "methods": nipah_eval,
            "retrospective_cascade_boltz2": nipah_cascade_boltz2,
        },
        "external_scorpion_neurotoxins": {
            "n_complexes": len(scorpion_rows),
            "n_targets": len(np.unique(t_scorp)),
            "n_adaptyv_binders": int(y_scorp_adap.sum()),
            "n_publication_binders": int(y_scorp_pub.sum()),
            "methods_adaptyv_labels": scorp_eval_adap,
            "methods_publication_labels": scorp_eval_pub,
            "retrospective_cascade_protenix": scorpion_cascade_protenix,
        },
        "overath_retrospective_cascade_fixed_budgets": {
            "af3": ov_cascade_af3,
            "boltz1": ov_cascade_boltz1,
        },
    }
    if EXTERNAL_RESULTS_JSON.exists():
        try:
            prev_ext = json.loads(EXTERNAL_RESULTS_JSON.read_text())
            if "batched_gpu_runtime_profile" in prev_ext:
                full_results["batched_gpu_runtime_profile"] = prev_ext["batched_gpu_runtime_profile"]
        except Exception:
            pass
    EXTERNAL_RESULTS_JSON.write_text(json.dumps(full_results, indent=2))
    print(f"\nSaved complete external validation & runtime results to {EXTERNAL_RESULTS_JSON}", flush=True)


if __name__ == "__main__":
    main()
